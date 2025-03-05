import os
import sys
import psycopg2
import json
from together import Together
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Together client
API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=API_KEY)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_PATH = "financial_data.index"
INDEX_MAP_PATH = "index_mapping.json"

# Connect to PostgreSQL
def get_db_connection():
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

# Lazy-load FAISS index
faiss_index = None
index_map = None

def load_faiss_index():
    global faiss_index, index_map
    if faiss_index is None or index_map is None:
        try:
            print(f"Loading FAISS index from: {INDEX_PATH}")
            faiss_index = faiss.read_index(INDEX_PATH)
            
            with open(INDEX_MAP_PATH, "r") as f:
                index_map = json.load(f)
            index_map = {int(k): v for k, v in index_map.items()}
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            faiss_index = None
            index_map = None

def build_faiss_index():
    """Builds the FAISS index only if it does not exist."""
    if os.path.exists(INDEX_PATH) and os.path.exists(INDEX_MAP_PATH):
        print("FAISS index already exists. Skipping rebuild.")
        return
    
    print("Building FAISS index...")

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Fetch category data first
        cur.execute('SELECT id, name FROM "Adviser_category"')
        category_data = cur.fetchall()
        category_dict = {id_: name for id_, name in category_data}  # {category_id: category_name}

        # Dictionary to store categorized data
        category_data_dict = defaultdict(lambda: {"transactions": [], "expenses": [], "budget": "No budget set"})

        # Fetch budget data
        cur.execute('SELECT id, category_id, name, "limit" FROM "BudgetTable"')
        budget_data = cur.fetchall()
        for _, category_id, name, limit in budget_data:
            category_name = category_dict.get(category_id, "Uncategorized")
            category_data_dict[category_name]["budget"] = f"Budget Limit: {limit}"

        # Fetch transaction data
        cur.execute('SELECT id, category_id, description, amount FROM "Adviser_transaction"')
        transaction_data = cur.fetchall()
        for _, category_id, description, amount in transaction_data:
            category_name = category_dict.get(category_id, "Uncategorized")
            transaction_text = f"Transaction: {description}, Transactions_Amount: {amount}"
            category_data_dict[category_name]["transactions"].append(transaction_text)

        # Fetch expense data
        cur.execute('SELECT id, category_id, name, amount FROM "Adviser_expense"')
        expense_data = cur.fetchall()
        for _, category_id, name, amount in expense_data:
            category_name = category_dict.get(category_id, "Uncategorized")
            expense_text = f"Expense: {name}, Expense_Amount: {amount}"
            category_data_dict[category_name]["expenses"].append(expense_text)

        # Merge all data
        all_data = []
        for category, details in category_data_dict.items():
            transactions_text = " | ".join(details["transactions"]) if details["transactions"] else "No transactions"
            expenses_text = " | ".join(details["expenses"]) if details["expenses"] else "No expenses"
            budget_text = details["budget"]

            # Calculate total amounts
            total_transactions_amount = sum(float(tx.split("Transactions_Amount: ")[1]) for tx in details["transactions"] if "Transactions_Amount: " in tx)
            total_expenses_amount = sum(float(exp.split("Expense_Amount: ")[1]) for exp in details["expenses"] if "Expense_Amount: " in exp)

            final_text = f"Category: {category}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}, Total Amount: {total_transactions_amount + total_expenses_amount}"
            all_data.append((category, final_text))

        # Ensure there's data
        if not all_data:
            raise ValueError("No data found in the database to build the index.")

        # Create embeddings
        descriptions = [str(item[1]) for item in all_data]
        embeddings = np.array([model.encode(desc) for desc in descriptions], dtype=np.float32)

        # Build and save FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)

        # Save index-to-ID mapping in JSON format
        index_mapping = {idx: id_ for idx, (id_, _) in enumerate(all_data)}
        with open(INDEX_MAP_PATH, "w") as f:
            json.dump(index_mapping, f)

        print("FAISS index built successfully.")

    finally:
        cur.close()
        conn.close()

# Query Meta-LLaMA 3.1 via Together AI
def query_llama3(prompt):
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

# Generate RAG response
def get_rag_response(query):
    global faiss_index, index_map
    
    # Ensure FAISS index is built and loaded
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEX_MAP_PATH):
        build_faiss_index()

    load_faiss_index()
    
    if faiss_index is None or index_map is None:
        return "Error: FAISS index not available."

    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=50)

    context = set()
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for idx in indices[0]:
            db_id = index_map.get(idx)
            if db_id:
                cur.execute('SELECT description, name, amount FROM "Adviser_transaction" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    description, name, amount = result
                    context.add(f"Transaction: {description}, Amount: {amount}, Category: {name}")

                cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    name, amount = result
                    context.add(f"Expense: {name}, Amount: {amount}")

                cur.execute('SELECT name, "limit" FROM "BudgetTable" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    name, limit = result
                    context.add(f"Budget: {name}, Limit: {limit}")

    finally:
        cur.close()
        conn.close()
    formatted_context = '\n'.join(context)  # Process separately
    prompt = f"Context:\n{formatted_context}\n\nQuestion:\n{query}\n\nAnswer:"

    return query_llama3(prompt)
