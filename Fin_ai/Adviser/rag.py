import os
import json
import psycopg2
import numpy as np
import faiss
from dotenv import load_dotenv
from together import Together
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Together AI client
API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=API_KEY)

# Initialize Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
# File paths for FAISS index
INDEX_PATH = "financial_data.index"
INDEX_MAP_PATH = "index_mapping.json"

# Lazy-load FAISS index and mapping
faiss_index = None
index_map = None


def get_db_connection():
    """Creates a PostgreSQL database connection."""
    return psycopg2.connect(
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
            else:
                print("FAISS index files not found. You may need to rebuild the index.")
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            faiss_index = None
            index_map = None


def build_faiss_index(batch_size=100):
    """Builds the FAISS index with batched processing to reduce memory usage."""
    if os.path.exists(INDEX_PATH) and os.path.exists(INDEX_MAP_PATH):
        print("FAISS index already exists. Skipping rebuild.")
        return

    print("Building FAISS index...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Fetch category data
        cur.execute('SELECT id, name FROM "Adviser_category"')
        category_dict = {id_: name for id_, name in cur.fetchall()}
        
        # Create index first, before adding data
        # Get dimension from model
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        
        # Initialize mapping dictionary
        index_mapping = {}
        current_idx = 0
        
        # Process each category separately to reduce memory usage
        for category_id, category_name in category_dict.items():
            # Create category context
            category_data = {"transactions": [], "expenses": [], "budget": "No budget set"}
            
            # Fetch budget
            cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = %s', (category_id,))
            budget = cur.fetchone()
            if budget:
                category_data["budget"] = f"Budget Limit: {budget[0]}"
            
            # Fetch transactions in batches
            cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = %s', (category_id,))
            while True:
                transactions = cur.fetchmany(batch_size)
                if not transactions:
                    break
                for desc, amount in transactions:
                    category_data["transactions"].append(f"Transaction: {desc}, Amount: {amount}")
            
            # Fetch expenses in batches
            cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = %s', (category_id,))
            while True:
                expenses = cur.fetchmany(batch_size)
                if not expenses:
                    break
                for name, amount in expenses:
                    category_data["expenses"].append(f"Expense: {name}, Amount: {amount}")
            
            # Create text chunks if there's too much data
            transactions_text = " | ".join(category_data["transactions"]) if category_data["transactions"] else "No transactions"
            expenses_text = " | ".join(category_data["expenses"]) if category_data["expenses"] else "No expenses"
            budget_text = category_data["budget"]
            
            # Process in smaller chunks if needed
            if len(category_data["transactions"]) > 20:
                # Process transactions in smaller groups to create multiple entries
                for i in range(0, len(category_data["transactions"]), 20):
                    chunk = category_data["transactions"][i:i+20]
                    chunk_text = f"Category: {category_name}, {budget_text}, Transactions: {' | '.join(chunk)}, Expenses: {expenses_text}"
                    
                    # Add to index
                    embedding = model.encode(chunk_text).reshape(1, -1).astype(np.float32)
                    index.add(embedding)
                    index_mapping[current_idx] = category_name
                    current_idx += 1
            else:
                # Create one entry for this category
                final_text = f"Category: {category_name}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}"
                embedding = model.encode(final_text).reshape(1, -1).astype(np.float32)
                index.add(embedding)
                index_mapping[current_idx] = category_name
                current_idx += 1
        
        # Save index and mapping
        faiss.write_index(index, INDEX_PATH)
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)
            
        print("FAISS index built successfully.")
    
    finally:
        cur.close()
        conn.close()

def query_llama3(prompt):
    """Queries Meta-LLaMA 3.1 via Together AI."""
    try:
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


def get_rag_response(query):
    """Retrieves relevant financial information using FAISS and queries LLaMA for an answer."""
    global faiss_index, index_map

    # Ensure FAISS index is available
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEX_MAP_PATH):
        build_faiss_index()

    load_faiss_index()

    if faiss_index is None or index_map is None:
        return "Error: FAISS index not available."

    # Encode query & search FAISS
    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=5)

    context = set()
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for idx in indices[0]:
            db_id = index_map.get(idx)
            if db_id:
                cur.execute('SELECT description, amount FROM "Adviser_transaction" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                for description, amount in cur.fetchall():
                    context.add(f"Transaction: {description}, Amount: {amount}")

                cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                for name, amount in cur.fetchall():
                    context.add(f"Expense: {name}, Amount: {amount}")

                cur.execute('SELECT "limit" FROM "BudgetTable" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                for limit in cur.fetchall():
                    context.add(f"Budget Limit: {limit[0]}")

    finally:
        cur.close()
        conn.close()

    formatted_context = '\n'.join(context)
    prompt = f"Context:\n{formatted_context}\n\nQuestion:\n{query}\n\nAnswer:"
    return query_llama3(prompt)