import os
import sys
import psycopg2
import json
from together import Together
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv

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
        # Fetch data from Adviser_transaction table
        cur.execute('SELECT id, name, description, amount FROM "Adviser_transaction"')
        transaction_data = cur.fetchall()
        
        # Fetch data from BudgetTable
        cur.execute('SELECT id, category_id, name, "limit" FROM "BudgetTable"')
        budget_data = cur.fetchall()
        
        # Fetch data from Adviser_category
        cur.execute('SELECT id, name FROM "Adviser_category"')
        category_data = cur.fetchall()

        # Create a mapping of category_id to category_name
        category_dict = {id_: name for id_, name in category_data}

        # Combine all data into a unified format
        # Combine transaction and budget data in a meaningful way
        all_data = []

        # Format transaction data with budget limits for better embeddings
        # Format transaction data with budget association
        for id_, name, description, amount in transaction_data:
            # Find matching budget category
            budget_limit = "No budget set"
            for budget_id, category_id, budget_name, limit in budget_data:
                if name.lower() == budget_name.lower():  # Match transaction name with budget category name
                    budget_limit = f"Budget Limit: {limit}"
                    break  # Stop searching once a match is found

            text = f"Transaction: {name}, Description: {description}, Amount Spent: {amount}, {budget_limit}"
            all_data.append((id_, text))

        # Format budget data independently (if needed)
        for id_, category_id, name, limit in budget_data:
            category_name = category_dict.get(category_id, "Unknown Category")
            text = f"Category: {name}, Budget Limit: {limit}"
            all_data.append((id_, text))

        if not all_data:
            raise ValueError("No data found in the database to build the index.")

        descriptions = [str(item[1]) for item in all_data]  # Extract text descriptions
        embeddings = np.array([model.encode(desc) for desc in descriptions], dtype=np.float32)


        # Print the embeddings and their corresponding text
        print("\nGenerated Embeddings:")
        for text, embedding in zip(descriptions, embeddings):
            print(f"Text: {text}")
            print(f"Embedding: {embedding[:5]} ... (truncated)\n")  # Print first 5 values for readability
            sys.stdout.flush()  # Force the output to be written immediately

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
    # Load FAISS index
    try:
        index = faiss.read_index("financial_data.index")
    except Exception as e:
        return f"Error loading FAISS index: {str(e)}"

    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)

    # Search for the top 5 similar entries
    distances, indices = index.search(query_embedding, k=11)

    # Retrieve context from both Adviser_transaction and BudgetTable
    context = []
    
    try:
        with open("index_mapping.json", "r") as f:
            index_map = json.load(f)
    except Exception as e:
        return f"Error loading index mapping: {str(e)}"

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        context = set()  # Change list to set to prevent duplicates

        for idx in indices[0]:
            db_id = index_map.get(str(idx))  # JSON keys are stored as strings
            if db_id is not None:
                cur.execute('SELECT description, name, amount FROM "Adviser_transaction" WHERE id = %s', (db_id,))
                result = cur.fetchone()
                if result:
                    description, name, amount = result
                    context.add(f"{description}, Amount: {amount}, Category: {name}")  # Use .add() to avoid duplicates

                cur.execute('SELECT name, "limit" FROM "BudgetTable" WHERE id = %s', (db_id,))
                result = cur.fetchone()
                if result:
                    name, limit = result
                    context.add(f"Budget Category: {name}, Limit: {limit}")  # Use .add() to avoid duplicates

    finally:
        cur.close()
        conn.close()

    # Construct prompt with context
    context_text = "\n".join(context)
    prompt = f"Given the following context, answer the question based solely on the provided information:\nContext: {context_text}\nQuestion: {query}\nAnswer:"
    
    # Query the model
    return query_llama3(prompt)

# Build the index if needed
if not os.path.exists("financial_data.index"):
    build_faiss_index()
