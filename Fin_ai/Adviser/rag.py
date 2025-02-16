import os
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
conn = psycopg2.connect(
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
cur = conn.cursor()

# Build FAISS index
def build_faiss_index():
    cur.execute('SELECT id, description FROM "Adviser_transaction"')
    data = cur.fetchall()
    if not data:
        raise ValueError("No data found in the database to build the index.")

    descriptions = [desc for _, desc in data]
    embeddings = np.array([model.encode(desc) for desc in descriptions], dtype=np.float32)

    # Build and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "financial_data.index")

    # Save index-to-ID mapping
    with open("index_mapping.txt", "w") as f:
        for idx, (id, _) in enumerate(data):
            f.write(f"{idx},{id}\n")

# Query Meta-LLaMA 3.1 via Together AI
def query_llama3(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5,
        top_p=0.9
    )
    return response.choices[0].message.content if response.choices else "No response received."

# Generate RAG response
def get_rag_response(query):
    # Load FAISS index
    index = faiss.read_index("financial_data.index")
    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)

    # Search for the top 5 similar entries
    distances, indices = index.search(query_embedding, k=5)

    # Retrieve context from the database
    context = []
    with open("index_mapping.txt", "r") as f:
        index_map = {int(line.split(",")[0]): int(line.split(",")[1]) for line in f}
    
    for idx in indices[0]:
        db_id = index_map.get(idx)
        if db_id is not None:
            cur.execute('SELECT description FROM "Adviser_transaction" WHERE id = %s', (db_id,))
            result = cur.fetchone()
            if result:
                context.append(result[0])

    # Construct prompt with context
    context_text = "\n".join(context)
    prompt = f"Context: {context_text}\nQuestion: {query}\nAnswer:"
    
    # Query the model
    return query_llama3(prompt)

# Build the index if needed
if not os.path.exists("financial_data.index"):
    build_faiss_index()

# Test the RAG pipeline
user_query = "How can I save more money?"
response = get_rag_response(user_query)
print(response)
