import os
import json
import psycopg2
import numpy as np
import gc
import tempfile
import mmap
import contextlib
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# File paths for vector storage
VECTORS_PATH = "financial_vectors.npz"
INDEX_MAP_PATH = "index_mapping.json"

# Install psutil if possible for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Memory monitoring function
def log_memory_usage(tag=""):
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage ({tag}): {memory_info.rss / 1024 / 1024:.2f} MB")
    else:
        print(f"Memory logging skipped - psutil not available ({tag})")

# Force garbage collection with memory logging
def force_gc(tag=""):
    gc.collect()
    log_memory_usage(tag)

@contextlib.contextmanager
def db_connection():
    """Single-use database connection context manager"""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD'),
            host=os.getenv('DATABASE_HOST'),
            port=os.getenv('DATABASE_PORT')
        )
        yield conn
    finally:
        if conn:
            conn.close()

def get_sentence_model():
    """Get the sentence transformer model - smallest possible"""
    # Using the absolute smallest model for embeddings
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

def get_together_client():
    """Get the Together client for LLM API access"""
    API_KEY = os.getenv("TOGETHER_API_KEY")
    return Together(api_key=API_KEY)

def process_category_chunk(category_ids, model):
    """Process a chunk of categories to create embeddings"""
    vectors = []
    category_names = []
    
    with db_connection() as conn:
        for category_id in category_ids:
            with conn.cursor() as cur:
                # Get category name
                cur.execute('SELECT name FROM "Adviser_category" WHERE id = %s', (category_id,))
                name_result = cur.fetchone()
                if not name_result:
                    continue
                
                category_name = name_result[0]
                
                # Build minimal context string
                context_parts = [f"Cat:{category_name[:20]}"]
                
                # Get budget (single query)
                cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s LIMIT 1', (category_id,))
                budget = cur.fetchone()
                if budget:
                    context_parts.append(f"Bud:{budget[0]}")
                
                # Get 3 transaction samples
                cur.execute(
                    'SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 3', 
                    (category_id,)
                )
                transactions = cur.fetchall()
                if transactions:
                    txns = "|".join(f"{d[:15]}:{a}" for d, a in transactions)
                    context_parts.append(f"Tx:{txns}")
                
                # Get 2 expense samples
                cur.execute(
                    'SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 2', 
                    (category_id,)
                )
                expenses = cur.fetchall()
                if expenses:
                    exps = "|".join(f"{n[:15]}:{a}" for n, a in expenses)
                    context_parts.append(f"Ex:{exps}")
                
                # Create compact text
                chunk_text = ";".join(context_parts)
                
                # Encode text
                embedding = model.encode(chunk_text, show_progress_bar=False).astype(np.float16)
                
                vectors.append(embedding)
                category_names.append(category_name)
    
    return vectors, category_names

def build_vector_index(micro_batch_size=3):
    """Ultra memory-efficient vector index builder using micro-batches and temp files"""
    log_memory_usage("start_build_index")
    print("Building vector index...")
    
    # Get all category IDs
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM "Adviser_category"')
            all_category_ids = [row[0] for row in cur.fetchall()]
    
    total_categories = len(all_category_ids)
    print(f"Total categories to process: {total_categories}")
    
    # Process in tiny batches and write to temp files to avoid memory buildup
    model = get_sentence_model()
    
    # Create a temporary file for storing vectors
    with tempfile.NamedTemporaryFile(delete=False) as temp_vector_file:
        vector_temp_path = temp_vector_file.name
    
    # Initialize index mapping
    index_mapping = {}
    current_idx = 0
    
    # Get embedding dimension for pre-allocation in each batch
    dimension = model.get_sentence_embedding_dimension()
    
    # Process in micro-batches
    for i in range(0, total_categories, micro_batch_size):
        batch_ids = all_category_ids[i:i+micro_batch_size]
        
        # Process this tiny batch
        vectors, names = process_category_chunk(batch_ids, model)
        
        if not vectors:
            continue
        
        # Update index mapping
        for j, name in enumerate(names):
            index_mapping[current_idx + j] = name
        
        # Convert to numpy array
        batch_array = np.array(vectors, dtype=np.float16)
        
        # Append to the NPZ file
        if os.path.exists(vector_temp_path) and os.path.getsize(vector_temp_path) > 0:
            # Load existing data
            with np.load(vector_temp_path, mmap_mode='r') as data:
                existing_vectors = data['vectors']
                combined = np.vstack([existing_vectors, batch_array])
                
                # Save combined data
                np.savez_compressed(vector_temp_path, vectors=combined)
                
                # Clear references
                del existing_vectors
                del combined
                force_gc(f"after_batch_{i}")
        else:
            # First batch
            np.savez_compressed(vector_temp_path, vectors=batch_array)
            
        # Update counter
        current_idx += len(vectors)
        
        # Cleanup batch resources
        del vectors
        del names
        del batch_array
        force_gc(f"processed_{current_idx}_of_{total_categories}")
        
        # Periodic logging
        if i % 10 == 0:
            print(f"Processed {i}/{total_categories} categories")
    
    # Move temp file to final location
    if os.path.exists(vector_temp_path):
        with open(vector_temp_path, 'rb') as src:
            with open(VECTORS_PATH, 'wb') as dst:
                dst.write(src.read())
        
        # Save mapping
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
            
        print(f"Vector index built successfully with {current_idx} vectors")
    
    # Cleanup temp file
    try:
        os.unlink(vector_temp_path)
    except:
        pass
    
    # Final cleanup
    del model
    del index_mapping
    force_gc("end_build_index")

@contextlib.contextmanager
def load_vector_chunk(start_idx, end_idx):
    """Load only a chunk of the vector index to conserve memory"""
    try:
        if os.path.exists(VECTORS_PATH):
            # Load only the specified slice using mmap for minimal memory impact
            with np.load(VECTORS_PATH, mmap_mode='r') as data:
                vectors = data['vectors'][start_idx:end_idx]
                yield vectors
        else:
            yield None
    finally:
        # Explicit cleanup
        force_gc("after_vector_chunk")

def cosine_similarity_single(query_vector, vector):
    """Calculate cosine similarity for a single vector pair"""
    query_norm = np.linalg.norm(query_vector)
    vec_norm = np.linalg.norm(vector)
    
    if query_norm == 0 or vec_norm == 0:
        return 0.0
    
    return np.dot(query_vector, vector) / (query_norm * vec_norm)

def find_top_matches(query_vector, chunk_size=100, top_k=3):
    """Find top matches using chunk-by-chunk processing to minimize memory usage"""
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(INDEX_MAP_PATH):
        return []
        
    # Get total vector count
    with np.load(VECTORS_PATH, mmap_mode='r') as data:
        total_vectors = len(data['vectors'])
    
    # Track top matches
    top_indices = []
    top_scores = []
    
    # Process vectors in small chunks
    for start_idx in range(0, total_vectors, chunk_size):
        end_idx = min(start_idx + chunk_size, total_vectors)
        
        with load_vector_chunk(start_idx, end_idx) as chunk:
            if chunk is None:
                continue
                
            # Process individual vectors to minimize memory
            for i in range(len(chunk)):
                score = cosine_similarity_single(query_vector, chunk[i])
                
                # Keep track of top matches
                if len(top_indices) < top_k:
                    top_indices.append(start_idx + i)
                    top_scores.append(score)
                else:
                    # Replace lowest score if better match found
                    min_idx = top_scores.index(min(top_scores))
                    if score > top_scores[min_idx]:
                        top_indices[min_idx] = start_idx + i
                        top_scores[min_idx] = score
        
        # Force cleanup after each chunk
        force_gc(f"processed_chunk_{start_idx}_{end_idx}")
    
    # Sort by score (descending)
    paired = sorted(zip(top_scores, top_indices), reverse=True)
    return [idx for _, idx in paired]

def query_llama3(prompt):
    """Minimal memory LLaMA query function"""
    try:
        client = get_together_client()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,  # Further reduced
            temperature=0.5,
            top_p=0.9
        )
        
        # Extract content and cleanup
        content = response.choices[0].message.content if response.choices else "No response received."
        del response
        
        return content
    except Exception as e:
        return f"Error in LLaMA API: {str(e)}"

def get_rag_response(query, max_context_items=5):  # Drastically reduced context
    """Ultra memory-efficient RAG implementation"""
    log_memory_usage("start_rag_response")
    
    # Ensure vector index exists
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(INDEX_MAP_PATH):
        print("Building index first...")
        build_vector_index()
    
    # Load index mapping (small JSON file)
    with open(INDEX_MAP_PATH, "r") as f:
        index_map = json.load(f)
    index_map = {int(k): v for k, v in index_map.items()}
    
    # Encode query with minimal memory impact
    model = get_sentence_model()
    query_embedding = model.encode(query, show_progress_bar=False).astype(np.float16)
    
    # Free model immediately
    del model
    force_gc("after_encoding")
    
    # Find top matches with minimal memory chunked processing
    top_indices = find_top_matches(query_embedding, chunk_size=50, top_k=2)  # Reduced from 3
    
    # Free embedding memory
    del query_embedding
    force_gc("after_search")
    
    # Collect minimal context
    context_items = []
    
    with db_connection() as conn:
        for idx in top_indices:
            if len(context_items) >= max_context_items:
                break
                
            db_id = index_map.get(str(idx))
            if not db_id:
                continue
            
            with conn.cursor() as cur:
                # Get category info
                cur.execute('SELECT id FROM "Adviser_category" WHERE name = %s', (db_id,))
                cat_result = cur.fetchone()
                if not cat_result:
                    continue
                
                category_id = cat_result[0]
                
                # Add category name
                context_items.append(f"Category: {db_id}")
                
                # Get budget (if space allows)
                if len(context_items) < max_context_items:
                    cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s LIMIT 1', (category_id,))
                    budget = cur.fetchone()
                    if budget:
                        context_items.append(f"Budget: {budget[0]}")
                
                # Get 2 transactions (if space allows)
                if len(context_items) < max_context_items:
                    cur.execute(
                        'SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 2', 
                        (category_id,)
                    )
                    for desc, amt in cur.fetchall():
                        if len(context_items) < max_context_items:
                            context_items.append(f"Transaction: {desc[:20]}, Amount: {amt}")
                
                # Get 1 expense (if space allows)
                if len(context_items) < max_context_items:
                    cur.execute(
                        'SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 1', 
                        (category_id,)
                    )
                    for name, amt in cur.fetchall():
                        if len(context_items) < max_context_items:
                            context_items.append(f"Expense: {name[:20]}, Amount: {amt}")
    
    # Create ultra-compact prompt
    context_text = '\n'.join(context_items)
    prompt = f"Context:\n{context_text}\n\nQ:{query}\n\nA:"
    
    # Free context memory
    del context_items
    del index_map
    force_gc("before_llm_call")
    
    # Query LLM
    response = query_llama3(prompt)
    
    # Final cleanup
    force_gc("end_rag_response")
    
    return response

def cleanup():
    """Full system cleanup"""
    # Force garbage collection
    for _ in range(3):  # Multiple passes
        gc.collect()
    
    print("Resources cleaned up")

if __name__ == "__main__":
    try:
        build_vector_index(micro_batch_size=2)  # Extremely small batches
    finally:
        cleanup()