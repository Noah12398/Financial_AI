import os
import json
import psycopg2
import numpy as np
import gc
from functools import lru_cache
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from psycopg2.pool import ThreadedConnectionPool
import weakref
import contextlib

# Load environment variables
load_dotenv()

# File paths for NumPy-based vector storage
VECTORS_PATH = "financial_vectors.npz"
INDEX_MAP_PATH = "index_mapping.json"

# Use weak references for global components
_together_client_ref = None
_sentence_model_ref = None
_connection_pool_ref = None

# Memory monitoring function
def log_memory_usage(tag=""):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage ({tag}): {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print(f"Memory logging skipped - psutil not available ({tag})")

@contextlib.contextmanager
def vector_index_context():
    """Context manager for loading and releasing vector index"""
    vector_index = None
    index_map = None
    
    try:
        if os.path.exists(VECTORS_PATH) and os.path.exists(INDEX_MAP_PATH):
            # Use memory mapping with read-only mode
            vector_index = np.load(VECTORS_PATH, mmap_mode='r')['vectors']
            with open(INDEX_MAP_PATH, "r") as f:
                index_map = json.load(f)
            index_map = {int(k): v for k, v in index_map.items()}
            print(f"NumPy vector index loaded successfully - {len(vector_index)} vectors")
        else:
            print("Vector index files not found. You may need to rebuild the index.")
            
        yield vector_index, index_map
    
    finally:
        # Explicitly release resources
        if vector_index is not None:
            del vector_index
        if index_map is not None:
            del index_map
        gc.collect()

def get_together_client():
    """Lazy initialization of Together AI client with weak reference"""
    global _together_client_ref
    
    client = None if _together_client_ref is None else _together_client_ref()
    
    if client is None:
        API_KEY = os.getenv("TOGETHER_API_KEY")
        client = Together(api_key=API_KEY)
        _together_client_ref = weakref.ref(client)
    
    return client

@lru_cache(maxsize=1)
def get_sentence_model():
    """Lazy initialization of sentence transformer model with caching"""
    # Use the smallest possible model for embedding
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

@contextlib.contextmanager
def db_connection():
    """Context manager for database connections"""
    global _connection_pool_ref
    
    pool = None if _connection_pool_ref is None else _connection_pool_ref()
    
    if pool is None:
        pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=3,  # Reduced from 5 to 3
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD'),
            host=os.getenv('DATABASE_HOST'),
            port=os.getenv('DATABASE_PORT'),
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        _connection_pool_ref = weakref.ref(pool)
    
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    finally:
        if conn is not None:
            pool.putconn(conn)

def batched_cosine_similarity(query_vector, vectors, batch_size=100):
    """Calculate cosine similarity between query vector and vectors in batches"""
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(len(vectors))
    
    query_normalized = query_vector / query_norm
    
    # Process in batches
    total_vectors = len(vectors)
    similarities = np.zeros(total_vectors, dtype=np.float32)
    
    for i in range(0, total_vectors, batch_size):
        end_idx = min(i + batch_size, total_vectors)
        batch = vectors[i:end_idx]
        
        # Compute norms for this batch
        batch_norms = np.linalg.norm(batch, axis=1)
        mask = batch_norms > 0
        
        # Initialize with zeros
        batch_similarities = np.zeros(end_idx - i, dtype=np.float32)
        
        # Calculate similarities only for non-zero vectors
        if np.any(mask):
            normalized_batch = batch[mask] / batch_norms[mask, np.newaxis]
            batch_similarities[mask] = np.dot(normalized_batch, query_normalized)
        
        similarities[i:end_idx] = batch_similarities
        
        # Clean up batch resources
        del batch, batch_norms, batch_similarities
    
    return similarities

def vector_search(query_vector, vectors, k=3):
    """Perform vector search with optimized memory usage"""
    # Use float32 for better memory efficiency without significant precision loss
    query_vector = query_vector.astype(np.float32)
    
    # Get similarities
    similarities = batched_cosine_similarity(query_vector, vectors)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Clean up
    del similarities
    
    return top_indices

def build_vector_index(batch_size=5):  # Reduced batch size
    """Builds the vector index with improved memory efficiency."""
    log_memory_usage("start_build_index")
    print("Building vector index...")
    
    # Get model dimension and prepare for smaller dtype
    model = get_sentence_model()
    
    # Use chunked processing
    chunks = []
    index_mapping = {}
    current_idx = 0
    
    with db_connection() as conn:
        # Get total count
        with conn.cursor() as count_cur:
            count_cur.execute('SELECT COUNT(*) FROM "Adviser_category"')
            total_categories = count_cur.fetchone()[0]
            print(f"Total categories to process: {total_categories}")
        
        # Process in smaller chunks for better memory management
        with conn.cursor(name='fetch_categories') as category_cur:
            category_cur.execute('SELECT id, name FROM "Adviser_category"')
            
            # Process in batches
            batch_count = 0
            batch_vectors = []
            batch_indices = []
            
            category_data = category_cur.fetchone()
            while category_data:
                category_id, category_name = category_data
                
                # Build a simplified context string
                context_parts = [f"Category:{category_name}"]
                
                # Get budget with dedicated cursor
                with conn.cursor() as budget_cur:
                    budget_cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
                    budget = budget_cur.fetchone()
                    if budget:
                        context_parts.append(f"Budget:{budget[0]}")
                
                # Get limited transactions (max 5)
                with conn.cursor(name=f'fetch_trans_{category_id}') as trans_cur:
                    trans_cur.execute(
                        'SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 5', 
                        (category_id,)
                    )
                    transactions = trans_cur.fetchall()
                    if transactions:
                        trans_text = "|".join(f"{d[:20]}:{a}" for d, a in transactions)  # Limit text length
                        context_parts.append(f"Trans:{trans_text}")
                
                # Get limited expenses (max 5)
                with conn.cursor(name=f'fetch_exp_{category_id}') as exp_cur:
                    exp_cur.execute(
                        'SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 5', 
                        (category_id,)
                    )
                    expenses = exp_cur.fetchall()
                    if expenses:
                        exp_text = "|".join(f"{n[:20]}:{a}" for n, a in expenses)  # Limit text length
                        context_parts.append(f"Exp:{exp_text}")
                
                # Create compact text representation
                chunk_text = ";".join(context_parts)
                
                # Encode text to vector - use float16 for memory efficiency
                embedding = model.encode(chunk_text, show_progress_bar=False).astype(np.float16)
                
                # Add to batch
                batch_vectors.append(embedding)
                batch_indices.append(category_name)
                
                # Process batch when it reaches the batch size
                if len(batch_vectors) >= batch_size:
                    # Get current chunk index
                    start_idx = current_idx
                    end_idx = start_idx + len(batch_vectors)
                    
                    # Update mapping
                    for i, name in enumerate(batch_indices):
                        index_mapping[start_idx + i] = name
                    
                    # Convert batch to numpy and add to chunks
                    batch_array = np.array(batch_vectors, dtype=np.float16)
                    chunks.append(batch_array)
                    
                    # Update counter and clear batch
                    current_idx = end_idx
                    batch_count += 1
                    batch_vectors = []
                    batch_indices = []
                    
                    # Force cleanup every few batches
                    if batch_count % 10 == 0:
                        gc.collect()
                        log_memory_usage(f"processed_{current_idx}_categories")
                
                # Get next category
                category_data = category_cur.fetchone()
        
        # Process any remaining items
        if batch_vectors:
            start_idx = current_idx
            for i, name in enumerate(batch_indices):
                index_mapping[start_idx + i] = name
            
            batch_array = np.array(batch_vectors, dtype=np.float16)
            chunks.append(batch_array)
        
        # Combine chunks efficiently
        total_vectors = sum(len(chunk) for chunk in chunks)
        print(f"Combining {len(chunks)} chunks with total {total_vectors} vectors")
        
        # Get dimension from first chunk
        vector_dim = chunks[0].shape[1] if chunks else 0
        
        # Pre-allocate final array
        final_array = np.empty((total_vectors, vector_dim), dtype=np.float16)
        
        # Copy chunks into final array
        current_pos = 0
        for chunk in chunks:
            chunk_size = len(chunk)
            final_array[current_pos:current_pos + chunk_size] = chunk
            current_pos += chunk_size
            del chunk  # Release memory immediately
        
        # Free chunks list
        chunks.clear()
        gc.collect()
        
        # Save compressed vectors
        np.savez_compressed(VECTORS_PATH, vectors=final_array)
        
        # Save mapping
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
        
        print(f"Vector index built successfully with {len(final_array)} vectors")
        
        # Cleanup
        del final_array
        del index_mapping
        gc.collect()
    
    log_memory_usage("end_build_index")

def query_llama3(prompt):
    """Queries Meta-LLaMA 3.1 via Together AI with memory-efficient approach."""
    try:
        client = get_together_client()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Reduced from 500
            temperature=0.5,
            top_p=0.9
        )
        return response.choices[0].message.content if response.choices else "No response received."
    except Exception as e:
        return f"Error in LLaMA API: {str(e)}"

def get_rag_response(query, max_context_items=10):  # Reduced from 25
    """Memory-optimized RAG implementation."""
    log_memory_usage("start_rag_response")
    
    # Ensure vector index is available
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(INDEX_MAP_PATH):
        build_vector_index()

    # Use context manager for vector index
    with vector_index_context() as (vector_index, index_map):
        if vector_index is None or index_map is None:
            return "Error: Vector index not available."

        # Encode query with efficient typing
        model = get_sentence_model()
        query_embedding = model.encode(query, show_progress_bar=False).astype(np.float16)
        
        # Perform vector search
        top_indices = vector_search(query_embedding, vector_index, k=3)
        
        # Free memory
        del query_embedding
        gc.collect()
        
        # Use a list instead of a set for better memory profile
        context_items = []
        
        # Use context manager for database connection
        with db_connection() as conn:
            for idx in top_indices:
                if len(context_items) >= max_context_items:
                    break
                    
                db_id = index_map.get(int(idx))
                if not db_id:
                    continue
                
                # Get transactions with strict limits
                with conn.cursor() as cur:
                    cur.execute('''
                        SELECT description, amount 
                        FROM "Adviser_transaction" 
                        WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                        LIMIT 5  -- Reduced from 20
                    ''', (db_id,))
                    
                    for description, amount in cur.fetchall():
                        if len(context_items) < max_context_items:
                            # Truncate description to save memory
                            context_items.append(f"Tx: {description[:30]}, Amt: {amount}")
                
                # Get expenses with strict limits
                with conn.cursor() as cur:
                    cur.execute('''
                        SELECT name, amount 
                        FROM "Adviser_expense" 
                        WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                        LIMIT 3  -- Reduced from 10
                    ''', (db_id,))
                    
                    for name, amount in cur.fetchall():
                        if len(context_items) < max_context_items:
                            # Truncate name to save memory
                            context_items.append(f"Exp: {name[:30]}, Amt: {amount}")
                
                # Get budget
                with conn.cursor() as cur:
                    cur.execute('''
                        SELECT "limit" 
                        FROM "BudgetTable" 
                        WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                        LIMIT 1
                    ''', (db_id,))
                    
                    limit = cur.fetchone()
                    if limit and len(context_items) < max_context_items:
                        context_items.append(f"Budget: {limit[0]}")
        
        # Create compact prompt
        formatted_context = '\n'.join(context_items)
        prompt = f"Context:\n{formatted_context}\n\nQ:{query}\n\nA:"
        
        # Free memory before LLM call
        del context_items
        gc.collect()
        
        log_memory_usage("before_llm_call")
        response = query_llama3(prompt)
        log_memory_usage("end_rag_response")
        
        return response

def cleanup():
    """Release all resources"""
    global _together_client_ref, _sentence_model_ref, _connection_pool_ref
    
    # Clear references
    _together_client_ref = None
    _sentence_model_ref = None
    
    # Close connection pool if exists
    pool = None if _connection_pool_ref is None else _connection_pool_ref()
    if pool is not None:
        pool.closeall()
    _connection_pool_ref = None
    
    # Clear LRU cache
    get_sentence_model.cache_clear()
    
    # Force garbage collection
    gc.collect()
    print("Resources cleaned up")

if __name__ == "__main__":
    try:
        build_vector_index()
    finally:
        cleanup()