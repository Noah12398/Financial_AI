import os
import json
import psycopg2
import numpy as np
import faiss
import gc
import time
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from psycopg2.pool import ThreadedConnectionPool

# Load environment variables
load_dotenv()

# File paths for FAISS index
INDEX_PATH = "financial_data.index"
INDEX_MAP_PATH = "index_mapping.json"
LAST_UPDATE_PATH = "last_update.json"

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
                return False
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            faiss_index = None
            index_map = None
            return False
    return True

def get_last_update_time():
    """Get the timestamp of the last index update"""
    if os.path.exists(LAST_UPDATE_PATH):
        try:
            with open(LAST_UPDATE_PATH, "r") as f:
                data = json.load(f)
                return data.get("last_update", 0)
        except:
            return 0
    return 0

def save_last_update_time():
    """Save the current timestamp as the last update time"""
    with open(LAST_UPDATE_PATH, "w") as f:
        json.dump({"last_update": time.time()}, f)

def check_for_updates(threshold_minutes=5):
    """Check if any database changes have occurred since last update
    Returns True if updates needed, False otherwise"""
    
    # If no index exists yet, we need a full build
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEX_MAP_PATH):
        return "full"
        
    last_update = get_last_update_time()
    # If it's been less than threshold_minutes since the last update, skip
    if time.time() - last_update < (threshold_minutes * 60):
        return False
        
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Check for new or modified transactions since last update
        cur.execute('''
            SELECT COUNT(*) FROM "Adviser_transaction" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        transaction_changes = cur.fetchone()[0]
        
        # Check for new or modified expenses since last update
        cur.execute('''
            SELECT COUNT(*) FROM "Adviser_expense" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        expense_changes = cur.fetchone()[0]
        
        # Check for new or modified budgets since last update
        cur.execute('''
            SELECT COUNT(*) FROM "BudgetTable" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        budget_changes = cur.fetchone()[0]
        
        # Check for new or modified categories since last update
        cur.execute('''
            SELECT COUNT(*) FROM "Adviser_category" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        category_changes = cur.fetchone()[0]
        
        cur.close()
        
        total_changes = transaction_changes + expense_changes + budget_changes + category_changes
        
        # If there are many changes, do a full rebuild
        if total_changes > 100 or category_changes > 0:
            return "full"
        # If there are some changes, do an incremental update
        elif total_changes > 0:
            return "incremental"
        # No changes
        return False
        
    except Exception as e:
        print(f"Error checking for updates: {str(e)}")
        # Fall back to using the index as is
        return False
    finally:
        release_db_connection(conn)

def update_faiss_index_incrementally():
    """Update the FAISS index with only changed data since last update"""
    global faiss_index, index_map
    
    log_memory_usage("start_incremental_update")
    
    # Load existing index
    if not load_faiss_index():
        # If loading fails, we need a full rebuild
        return build_faiss_index()
        
    last_update = get_last_update_time()
    model = get_sentence_model()
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        
        # Get changed categories
        cur.execute('''
            SELECT id, name FROM "Adviser_category" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        changed_categories = cur.fetchall()
        
        # If categories changed, we need a full rebuild (simplest approach)
        if changed_categories:
            cur.close()
            release_db_connection(conn)
            return build_faiss_index()
            
        # Get categories with changed transactions, expenses or budgets
        cur.execute('''
            SELECT DISTINCT category_id FROM "Adviser_transaction" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        changed_transaction_categories = [row[0] for row in cur.fetchall()]
        
        cur.execute('''
            SELECT DISTINCT category_id FROM "Adviser_expense" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        changed_expense_categories = [row[0] for row in cur.fetchall()]
        
        cur.execute('''
            SELECT DISTINCT category_id FROM "BudgetTable" 
            WHERE updated_at > to_timestamp(%s)
        ''', (last_update,))
        changed_budget_categories = [row[0] for row in cur.fetchall()]
        
        # Combine all changed category IDs
        changed_category_ids = set(changed_transaction_categories + 
                                 changed_expense_categories + 
                                 changed_budget_categories)
        
        if not changed_category_ids:
            print("No changes detected for incremental update")
            cur.close()
            release_db_connection(conn)
            return False
            
        # Get category names from IDs
        changed_category_names = {}
        for cat_id in changed_category_ids:
            cur.execute('SELECT name FROM "Adviser_category" WHERE id = %s', (cat_id,))
            result = cur.fetchone()
            if result:
                changed_category_names[cat_id] = result[0]
        
        # Find embeddings to remove
        embeddings_to_remove = []
        for idx, cat_name in index_map.items():
            if cat_name in changed_category_names.values():
                embeddings_to_remove.append(int(idx))
        
        # Create new index without these embeddings
        dimension = model.get_sentence_embedding_dimension()
        new_index = faiss.IndexFlatL2(dimension)
        new_index_map = {}
        
        # Copy unaffected embeddings
        if embeddings_to_remove:
            for i in range(faiss_index.ntotal):
                if i not in embeddings_to_remove:
                    # Extract the embedding
                    embedding = faiss.vector_float_to_array(faiss_index.get_vector(i))
                    embedding = embedding.reshape(1, -1).astype(np.float32)
                    # Add to new index
                    new_index.add(embedding)
                    # Update mapping
                    new_index_map[len(new_index_map)] = index_map[str(i)]
        else:
            # No embeddings to remove, just use the existing index
            new_index = faiss_index
            new_index_map = {str(k): v for k, v in index_map.items()}
                    
        # Process each changed category
        next_idx = new_index.ntotal
        
        for category_id, category_name in changed_category_names.items():
            # Create category context
            transactions = []
            expenses = []
            budget_text = "No budget set"
            
            # Fetch budget
            cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
            budget = cur.fetchone()
            if budget:
                budget_text = f"Budget Limit: {budget[0]}"
            
            # Get transactions in small batches
            cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s LIMIT 100', (category_id,))
            transactions = [f"Transaction: {desc}, Amount: {amount}" for desc, amount in cur.fetchall()]
            
            # Get expenses
            cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 50', (category_id,))
            expenses = [f"Expense: {name}, Amount: {amount}" for name, amount in cur.fetchall()]
            
            # Create chunks if there are many transactions
            if len(transactions) > 20:
                # Process in chunks of 20
                for i in range(0, len(transactions), 20):
                    chunk = transactions[i:i+20]
                    expenses_text = " | ".join(expenses) if expenses else "No expenses"
                    transactions_text = " | ".join(chunk)
                    
                    # Create chunk text and add to index
                    chunk_text = f"Category: {category_name}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}"
                    embedding = model.encode(chunk_text).reshape(1, -1).astype(np.float32)
                    new_index.add(embedding)
                    new_index_map[next_idx] = category_name
                    next_idx += 1
                    
                    # Force garbage collection
                    del embedding
                    gc.collect()
            else:
                # Just one chunk for this category
                expenses_text = " | ".join(expenses) if expenses else "No expenses"
                transactions_text = " | ".join(transactions) if transactions else "No transactions"
                
                chunk_text = f"Category: {category_name}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}"
                embedding = model.encode(chunk_text).reshape(1, -1).astype(np.float32)
                new_index.add(embedding)
                new_index_map[next_idx] = category_name
                next_idx += 1
                
                # Force garbage collection
                del embedding
                gc.collect()
            
            # Clear variables to free memory
            del transactions
            del expenses
            gc.collect()
            
        # Save the updated index
        faiss.write_index(new_index, INDEX_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(new_index_map, f)
            
        # Update globals
        faiss_index = new_index
        index_map = {int(k): v for k, v in new_index_map.items()}
        
        # Update last update time
        save_last_update_time()
        
        print(f"FAISS index updated incrementally with {len(changed_category_ids)} categories")
        
    except Exception as e:
        print(f"Error in incremental update: {str(e)}")
        # Fall back to full rebuild on error
        return build_faiss_index()
    finally:
        cur.close()
        release_db_connection(conn)
        gc.collect()
        log_memory_usage("end_incremental_update")
    
    return True

def build_faiss_index(batch_size=30):  # Further reduced batch size
    """Builds the FAISS index with efficient memory usage."""
    log_memory_usage("start_build_index")
    print("Building FAISS index from scratch...")
    
    # Free any existing resources
    cleanup()
    
    # Get model dimension once
    model = get_sentence_model()
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    index_mapping = {}
    current_idx = 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch category IDs first, but process one at a time
        cur.execute('SELECT id, name FROM "Adviser_category"')
        categories = cur.fetchall()
        
        # Process categories in smaller batches to reduce memory usage
        batch_size = min(5, len(categories))  # Process max 5 categories at a time
        
        for i in range(0, len(categories), batch_size):
            category_batch = categories[i:i+batch_size]
            
            for category_id, category_name in category_batch:
                # Create category context - use list only for current processing
                transactions = []
                expenses = []
                budget_text = "No budget set"
                
                # Fetch budget
                cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
                budget = cur.fetchone()
                if budget:
                    budget_text = f"Budget Limit: {budget[0]}"
                
                # Process transactions in smaller batches (20 at a time)
                chunk_size = 20
                cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s', (category_id,))
                
                # Process transactions in chunks and index directly
                while True:
                    transactions = []
                    batch = cur.fetchmany(chunk_size)
                    if not batch:
                        break
                    
                    for desc, amount in batch:
                        transactions.append(f"Transaction: {desc}, Amount: {amount}")
                    
                    # Get expenses once per transaction batch for this category
                    if not expenses:
                        expenses_cur = conn.cursor()
                        expenses_cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s LIMIT 20', (category_id,))
                        expenses = [f"Expense: {name}, Amount: {amount}" for name, amount in expenses_cur.fetchall()]
                        expenses_cur.close()
                    
                    # Create text representation
                    expenses_text = " | ".join(expenses) if expenses else "No expenses"
                    transactions_text = " | ".join(transactions) if transactions else "No transactions"
                    
                    # Create chunk text and add to index
                    chunk_text = f"Category: {category_name}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}"
                    embedding = model.encode(chunk_text).reshape(1, -1).astype(np.float32)
                    index.add(embedding)
                    index_mapping[current_idx] = category_name
                    current_idx += 1
                    
                    # Force garbage collection after each embedding
                    del embedding
                    gc.collect()
                
                # Clear category data before moving to next category
                del transactions
                del expenses
                gc.collect()
                
                print(f"Processed category: {category_name}")
            
            # Force garbage collection after each batch of categories
            gc.collect()
            log_memory_usage(f"after_category_batch_{i//batch_size}")
        
        # Save index and mapping
        faiss.write_index(index, INDEX_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
        
        # Update global variables
        global faiss_index, index_map
        faiss_index = index
        index_map = {int(k): v for k, v in index_mapping.items()}
        
        # Update last update time
        save_last_update_time()
            
        print("FAISS index built successfully.")
        return True
    
    except Exception as e:
        print(f"Error building index: {str(e)}")
        raise
    
    finally:
        cur.close()
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

def get_rag_response(query, max_context_items=50):  # Reduced context items
    """Retrieves relevant financial information using FAISS and queries LLaMA for an answer."""
    log_memory_usage("start_rag_response")
    
    # Check if we need to update the index (but don't block the query with a full rebuild)
    update_status = check_for_updates()
    if update_status == "incremental":
        print("Running incremental update before query")
        update_faiss_index_incrementally()
    # If full rebuild needed but not urgent, schedule it separately
    elif update_status == "full":
        print("Full index rebuild needed - will use existing index for now")
    
    # Ensure FAISS index is loaded
    load_faiss_index()

    if faiss_index is None or index_map is None:
        return "Error: FAISS index not available."

    # Encode query & search FAISS - use smaller k value
    model = get_sentence_model()
    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=2)  # Further reduced from 3 to 2

    # Free memory
    del query_embedding
    gc.collect()

    context = set()
    conn = get_db_connection()
    
    try:
        for idx in indices[0]:
            if len(context) >= max_context_items:
                break
                
            db_id = index_map.get(int(idx))
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
                    LIMIT 15
                ''', (db_id,))
                
                for description, amount in cur.fetchall():
                    if len(context) < max_context_items:
                        context.add(f"Transaction: {description}, Amount: {amount}")
                
                cur.execute('''
                    SELECT name, amount 
                    FROM "Adviser_expense" 
                    WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)
                    LIMIT 5
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
    # For initial setup
    build_faiss_index()