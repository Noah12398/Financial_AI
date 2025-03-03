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

def build_faiss_index():
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

        # Print debug info
        print("\nGenerated Embeddings:")
        for text, embedding in zip(descriptions, embeddings):
            print(f"Text: {text}")
            print(f"Embedding: {embedding[:5]} ... (truncated)\n")
            sys.stdout.flush()

        # Build and save FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "financial_data.index")

        # Save index-to-ID mapping in JSON format
        index_mapping = {idx: id_ for idx, (id_, _) in enumerate(all_data)}
        with open("index_mapping.json", "w") as f:
            json.dump(index_mapping, f)

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
    try:
        
        index = faiss.read_index("financial_data.index")
    except Exception as e:
        return f"Error loading FAISS index: {str(e)}"

    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_embedding, k=15)

    context = set()

    try:
        with open("index_mapping.json", "r") as f:
            index_map = json.load(f)
        index_map = {int(k): v for k, v in index_map.items()}
    except Exception as e:
        return f"Error loading index mapping: {str(e)}"
    # print("ASSDDFF")
    # print("FAISS Retrieved IDs:", indices[0])
    # print("Mapped DB IDs:", [index_map.get(idx) for idx in indices[0]]) 

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for idx in indices[0]:
            db_id = index_map.get(idx)  # Correct

            print(db_id)
            total_transactions_amounts=0
            total_expenses_amounts=0
            if db_id is not None:
                # Fetch Transactions
                cur.execute('SELECT description, amount FROM "Adviser_transaction"  WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    description, amount = result
                    total_transactions_amounts=amount
                    context.add(f"Transaction: {description}, Amount: {amount}")

                # Fetch Expenses
                cur.execute('SELECT name, amount FROM "Adviser_expense" WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    name, amount = result
                    total_expenses_amounts=amount
                    context.add(f"Expense: {name}, Amount: {amount}")

                # Fetch Budgets
                cur.execute('SELECT name, "limit" FROM "BudgetTable"  WHERE category_id = (SELECT id FROM "Adviser_category" WHERE name = %s)', (db_id,))
                result = cur.fetchone()
                if result:
                    name, limit = result
                    context.add(f"Budget Category: {name}, Limit: {limit}")
                context.add(f"Total Amount: {total_transactions_amounts + total_expenses_amounts}")
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

    finally:
        cur.close()
        conn.close()
    print(context)
    context_text = "\n".join(context)
    prompt = f"""
    You are a financial AI assistant. Based on the given financial data, analyze and respond accordingly.

    Context:
    {context_text}

    Question:
    {query}

    Answer:
    """

    return query_llama3(prompt)

# Build the index if needed
if not os.path.exists("financial_data.index"):
    build_faiss_index()