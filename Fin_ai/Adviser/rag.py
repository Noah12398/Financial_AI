import os
import json
import psycopg2
import numpy as np
import faiss
import gc
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from psycopg2.pool import ThreadedConnectionPool

# Load environment variables
load_dotenv()

# File paths for FAISS index
INDEX_PATH = "financial_data.index"
INDEX_MAP_PATH = "index_mapping.json"

# Initialize necessary components only when needed
together_client = None
sentence_model = None
faiss_index = None
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

def load_faiss_index():
    """Loads the FAISS index and mapping if not already loaded."""
    global faiss_index, index_map

    if faiss_index is None or index_map is None:
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(INDEX_MAP_PATH):
                faiss_index = faiss.read_index(INDEX_PATH)
                with open(INDEX_MAP_PATH, "r") as f:
                    index_map = json.load(f)
                index_map = {int(k): v for k, v in index_map.items()}
                print("FAISS index loaded successfully")
            else:
                print("FAISS index files not found. You may need to rebuild the index.")
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            faiss_index = None
            index_map = None

def build_faiss_index(batch_size=20):  # Smaller batch size for better memory management
    """Builds the FAISS index with improved memory efficiency."""
    log_memory_usage("start_build_index")
    print("Building FAISS index...")
    
    # Use IVFFlat index which is more memory efficient for larger datasets
    # Get model dimension once
    model = get_sentence_model()
    dimension = model.get_sentence_embedding_dimension()
    
    # Create a more memory-efficient index type
    # IVFFlat with fewer centroids is more memory efficient
    nlist = 10  # fewer clusters for small dataset
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Initialize with some data before adding vectors
    index_mapping = {}
    current_idx = 0
    
    conn = get_db_connection()
    
    try:
        # Create a faiss batch matrix that we'll reuse to avoid allocation/deallocation cycles
        max_batch_vectors = batch_size
        batch_embeddings = np.zeros((max_batch_vectors, dimension), dtype=np.float32)
        batch_count = 0
        batch_indices = []  # Store corresponding category names
        
        # First get category count to determine training samples
        with conn.cursor() as count_cur:
            count_cur.execute('SELECT COUNT(*) FROM "Adviser_category"')
            category_count = count_cur.fetchone()[0]
        
        # Create a separate cursor for the main query
        with conn.cursor(name='fetch_categories') as category_cur:  # Named cursor for server-side cursor
            # Use server-side cursor to avoid fetching all records at once
            category_cur.execute('SELECT id, name FROM "Adviser_category"')
            
            # We need some vectors to train the IVF index
            train_vectors = []
            
            # Process first few categories to get training vectors
            train_limit = min(100, category_count)
            for i in range(train_limit):
                category_data = category_cur.fetchone()
                if not category_data:
                    break
                
                category_id, category_name = category_data
                
                # Create a simplified representation for training
                train_text = f"Category: {category_name}"
                train_embedding = model.encode(train_text).reshape(1, -1).astype(np.float32)
                train_vectors.append(train_embedding)
            
            # Combine training vectors and train the index
            if train_vectors:
                train_data = np.vstack(train_vectors)
                index.train(train_data)
                
                # Free training data memory
                del train_vectors
                del train_data
                gc.collect()
            
            # Reset cursor to start and process categories one by one
            category_cur.execute('SELECT id, name FROM "Adviser_category"')
            
            # Process each category separately 
            while True:
                category_data = category_cur.fetchone()
                if not category_data:
                    break
                    
                category_id, category_name = category_data
                
                # Clear per-category variables to free memory
                budget_text = "No budget set"
                
                # Get budget info with its own connection to avoid memory leaks
                with conn.cursor() as budget_cur:
                    budget_cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
                    budget = budget_cur.fetchone()
                    if budget:
                        budget_text = f"Budget Limit: {budget[0]}"
                
                # Process transactions in smaller batches
                transactions_text = ""
                with conn.cursor(name=f'fetch_trans_{category_id}') as trans_cur:
                    trans_cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 10', (category_id,))
                    transactions = trans_cur.fetchall()
                    
                    if transactions:
                        transaction_items = [f"{desc}:{amt}" for desc, amt in transactions]
                        transactions_text = " | ".join(transaction_items)
                    else:
                        transactions_text = "No transactions"
                
                # Get expenses with its own cursor
                expenses_text = ""
                with conn.cursor(name=f'fetch_exp_{category_id}') as exp_cur:
                    exp_cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 10', (category_id,))
                    expenses = exp_cur.fetchall()
                    
                    if expenses:
                        expense_items = [f"{name}:{amt}" for name, amt in expenses]
                        expenses_text = " | ".join(expense_items)
                    else:
                        expenses_text = "No expenses"
                
                # Create category representation with minimal text
                chunk_text = f"Category:{category_name};Budget:{budget_text};Transactions:{transactions_text};Expenses:{expenses_text}"
                
                # Add to batch
                embedding = model.encode(chunk_text).astype(np.float32)
                batch_embeddings[batch_count] = embedding
                batch_indices.append(category_name)
                batch_count += 1
                
                # Add completed batch to index
                if batch_count >= max_batch_vectors:
                    active_batch = batch_embeddings[:batch_count]
                    index.add(active_batch)
                    
                    # Update mapping
                    for i, name in enumerate(batch_indices):
                        index_mapping[current_idx + i] = name
                    
                    current_idx += batch_count
                    batch_count = 0
                    batch_indices = []
                
                # Force garbage collection periodically
                if current_idx % 100 == 0:
                    gc.collect()
                    log_memory_usage(f"processed_{current_idx}_categories")
        
        # Add any remaining batch items
        if batch_count > 0:
            active_batch = batch_embeddings[:batch_count]
            index.add(active_batch)
            
            # Update mapping
            for i, name in enumerate(batch_indices):
                index_mapping[current_idx + i] = name
        
        # Save index and mapping
        faiss.write_index(index, INDEX_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
            
        print("FAISS index built successfully with improved memory efficiency.")
    
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
    """Retrieves relevant financial information using FAISS and queries LLaMA for an answer."""
    log_memory_usage("start_rag_response")
    
    # Ensure FAISS index is available
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEX_MAP_PATH):
        build_faiss_index()

    load_faiss_index()

    if faiss_index is None or index_map is None:
        return "Error: FAISS index not available."

    # Encode query & search FAISS - use smaller k value
    model = get_sentence_model()
    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=3)  # Reduced from 5 to 3

    # Free memory
    del query_embedding
    gc.collect()

    context = set()
    conn = get_db_connection()
    
    try:
        for idx in indices[0]:
            if len(context) >= max_context_items:
                break
                
            db_id = index_map.get(idx)
            if not db_id:
                continue
                
            # Use separate cursor for each query
            cur = conn.cursor()
            try:
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
                cur.close()

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
    global sentence_model, faiss_index, index_map, connection_pool
    
    # Release model
    if sentence_model is not None:
        del sentence_model
        sentence_model = None
    
    # Release FAISS resources
    if faiss_index is not None:
        del faiss_index
        faiss_index = None
    
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
    build_faiss_index()