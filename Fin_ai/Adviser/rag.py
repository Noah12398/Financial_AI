import os
import json
import psycopg2
import numpy as np
import gc
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from psycopg2.pool import ThreadedConnectionPool

# Load environment variables
load_dotenv()

# File paths for NumPy-based vector storage
VECTORS_PATH = "financial_vectors.npz"
INDEX_MAP_PATH = "index_mapping.json"

# Initialize necessary components only when needed
together_client = None
sentence_model = None
vector_index = None  # Will store the NumPy array of vectors
index_map = None
connection_pool = None

# Memory monitoring function
def log_memory_usage(tag=""):
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage ({tag}): {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print(f"Memory logging skipped - psutil not available ({tag})")

def get_together_client():
    """Lazy initialization of Together AI client"""
    global together_client
    if together_client is None:
        API_KEY = os.getenv("TOGETHER_API_KEY")
        together_client = Together(api_key=API_KEY)
    return together_client

def get_sentence_model():
    """Lazy initialization of sentence transformer model"""
    global sentence_model
    if sentence_model is None:
        # Use a smaller model if available for memory efficiency
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return sentence_model

def get_connection_pool():
    """Lazy initialization of database connection pool"""
    global connection_pool
    if connection_pool is None:
        connection_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=5,  # Limit concurrent connections
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
    return connection_pool

def get_db_connection():
    """Get a connection from the pool"""
    return get_connection_pool().getconn()

def release_db_connection(conn):
    """Release connection back to the pool"""
    if conn and connection_pool:
        get_connection_pool().putconn(conn)

def load_vector_index():
    """Loads the NumPy-based vector index and mapping if not already loaded."""
    global vector_index, index_map

    if vector_index is None or index_map is None:
        try:
            if os.path.exists(VECTORS_PATH) and os.path.exists(INDEX_MAP_PATH):
                npz_data = np.load(VECTORS_PATH)
                vector_index = npz_data['vectors']
                with open(INDEX_MAP_PATH, "r") as f:
                    index_map = json.load(f)
                index_map = {int(k): v for k, v in index_map.items()}
                print(f"NumPy vector index loaded successfully - {len(vector_index)} vectors")
            else:
                print("Vector index files not found. You may need to rebuild the index.")
        except Exception as e:
            print(f"Error loading vector index: {str(e)}")
            vector_index = None
            index_map = None

def cosine_similarity(query_vector, vectors):
    """Calculate cosine similarity between query vector and a matrix of vectors"""
    # Normalize vectors for cosine similarity
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(len(vectors))
    
    query_normalized = query_vector / query_norm
    
    # Calculate dot product, handling empty vectors
    similarities = np.zeros(len(vectors))
    for i, vec in enumerate(vectors):
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            similarities[i] = np.dot(query_normalized, vec / vec_norm)
    
    return similarities

def vector_search(query_vector, vectors, k=3):
    """Perform vector search using NumPy and return top k indices"""
    # Calculate similarities
    similarities = cosine_similarity(query_vector, vectors)
    
    # Get top k indices
    if len(similarities) <= k:
        return np.argsort(similarities)[::-1]
    else:
        return np.argsort(similarities)[::-1][:k]

def build_vector_index(batch_size=20):
    """Builds the vector index with improved memory efficiency using NumPy."""
    log_memory_usage("start_build_index")
    print("Building vector index...")
    
    # Get model dimension
    model = get_sentence_model()
    dimension = model.get_sentence_embedding_dimension()
    
    # Vectors will be stored in this list initially
    all_vectors = []
    index_mapping = {}
    current_idx = 0
    
    conn = get_db_connection()
    
    try:
        # Use a connection to get the total count for pre-allocation
        with conn.cursor() as count_cur:
            count_cur.execute('SELECT COUNT(*) FROM "Adviser_category"')
            total_categories = count_cur.fetchone()[0]
            print(f"Total categories to process: {total_categories}")
        
        # Pre-allocate vectors if we know the size (better memory management)
        # Will use a list first, then convert to numpy array at the end
        
        # Process in batches
        batch_vectors = []
        batch_indices = []
        
        # Process each category separately with a server-side cursor
        with conn.cursor(name='fetch_categories') as category_cur:
            category_cur.execute('SELECT id, name FROM "Adviser_category"')
            
            # Process each category separately
            while True:
                category_data = category_cur.fetchone()
                if not category_data:
                    break
                    
                category_id, category_name = category_data
                
                # Get budget info with dedicated cursor
                budget_text = "No budget set"
                with conn.cursor() as budget_cur:
                    budget_cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
                    budget = budget_cur.fetchone()
                    if budget:
                        budget_text = f"Budget Limit: {budget[0]}"
                
                # Get transactions with dedicated cursor - limit to 10
                transactions_text = ""
                with conn.cursor(name=f'fetch_trans_{category_id}') as trans_cur:
                    trans_cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 10', (category_id,))
                    transactions = trans_cur.fetchall()
                    
                    if transactions:
                        transaction_items = [f"{desc}:{amt}" for desc, amt in transactions]
                        transactions_text = " | ".join(transaction_items)
                    else:
                        transactions_text = "No transactions"
                
                # Get expenses with dedicated cursor
                expenses_text = ""
                with conn.cursor(name=f'fetch_exp_{category_id}') as exp_cur:
                    exp_cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 10', (category_id,))
                    expenses = exp_cur.fetchall()
                    
                    if expenses:
                        expense_items = [f"{name}:{amt}" for name, amt in expenses]
                        expenses_text = " | ".join(expense_items)
                    else:
                        expenses_text = "No expenses"
                
                # Create minimal text representation
                chunk_text = f"Category:{category_name};Budget:{budget_text};Transactions:{transactions_text};Expenses:{expenses_text}"
                
                # Encode text to vector
                embedding = model.encode(chunk_text).astype(np.float32)
                
                # Add to batch
                batch_vectors.append(embedding)
                batch_indices.append(category_name)
                
                # Process batch
                if len(batch_vectors) >= batch_size:
                    # Add vectors to main collection
                    all_vectors.extend(batch_vectors)
                    
                    # Update mapping
                    for i, name in enumerate(batch_indices):
                        index_mapping[current_idx + i] = name
                    
                    current_idx += len(batch_vectors)
                    batch_vectors = []
                    batch_indices = []
                    
                    # Memory cleanup for large datasets
                    if current_idx % 100 == 0:
                        gc.collect()
                        log_memory_usage(f"processed_{current_idx}_categories")
        
        # Add any remaining batch items
        if batch_vectors:
            all_vectors.extend(batch_vectors)
            for i, name in enumerate(batch_indices):
                index_mapping[current_idx + i] = name
        
        # Convert list of vectors to numpy array
        vector_array = np.array(all_vectors, dtype=np.float32)
        print(f"Created vector array with shape: {vector_array.shape}")
        
        # Save vector array and mapping
        np.savez_compressed(VECTORS_PATH, vectors=vector_array)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
            
        print("Vector index built successfully.")
        
        # Cleanup
        del all_vectors
        del vector_array
        gc.collect()
    
    except Exception as e:
        print(f"Error building index: {str(e)}")
        raise
    
    finally:
        release_db_connection(conn)
        # Force garbage collection
        gc.collect()
        log_memory_usage("end_build_index")

def query_llama3(prompt):
    """Queries Meta-LLaMA 3.1 via Together AI."""
    try:
        client = get_together_client()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5,
            top_p=0.9
        )
        return response.choices[0].message.content if response.choices else "No response received."
    except Exception as e:
        return f"Error in LLaMA API: {str(e)}"

def get_rag_response(query, max_context_items=100):
    """Retrieves relevant financial information using NumPy-based vector search and queries LLaMA for an answer."""
    log_memory_usage("start_rag_response")
    
    # Ensure vector index is available
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(INDEX_MAP_PATH):
        build_vector_index()

    load_vector_index()

    if vector_index is None or index_map is None:
        return "Error: Vector index not available."

    # Encode query & search vectors
    model = get_sentence_model()
    query_embedding = model.encode(query).astype(np.float32)
    
    # Perform vector search
    top_indices = vector_search(query_embedding, vector_index, k=3)

    # Free memory
    del query_embedding
    gc.collect()

    context = set()
    conn = get_db_connection()
    
    try:
        for idx in top_indices:
            if len(context) >= max_context_items:
                break
                
            db_id = index_map.get(int(idx))
            if not db_id:
                continue
                
            # Use with statement for better resource management
            with conn.cursor() as cur:
                # Limit results to avoid memory issues
                cur.execute('''
                    SELECT description, amount 
                    FROM "Adviser_transaction" 
                    WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                    LIMIT 20
                ''', (db_id,))
                
                for description, amount in cur.fetchall():
                    if len(context) < max_context_items:
                        context.add(f"Transaction: {description}, Amount: {amount}")
                
                cur.execute('''
                    SELECT name, amount 
                    FROM "Adviser_expense" 
                    WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                    LIMIT 10
                ''', (db_id,))
                
                for name, amount in cur.fetchall():
                    if len(context) < max_context_items:
                        context.add(f"Expense: {name}, Amount: {amount}")
                
                cur.execute('''
                    SELECT "limit" 
                    FROM "BudgetTable" 
                    WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                    LIMIT 1
                ''', (db_id,))
                
                limit = cur.fetchone()
                if limit and len(context) < max_context_items:
                    context.add(f"Budget Limit: {limit[0]}")

    finally:
        release_db_connection(conn)

    # Format context more efficiently
    formatted_context = '\n'.join(context)
    prompt = f"Context:\n{formatted_context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    # Free memory before LLM call
    del context
    gc.collect()
    
    log_memory_usage("before_llm_call")
    response = query_llama3(prompt)
    log_memory_usage("end_rag_response")
    
    return response

# Clean up function to release resources
def cleanup():
    """Release all resources"""
    global sentence_model, vector_index, index_map, connection_pool
    
    # Release model
    if sentence_model is not None:
        del sentence_model
        sentence_model = None
    
    # Release vector resources
    if vector_index is not None:
        del vector_index
        vector_index = None
    
    if index_map is not None:
        del index_map
        index_map = None
    
    # Close connection pool
    if connection_pool is not None:
        connection_pool.closeall()
        connection_pool = None
    
    # Force garbage collection
    gc.collect()
    print("Resources cleaned up")

if __name__ == "__main__":
    build_vector_index()