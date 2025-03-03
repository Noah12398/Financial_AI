import psycopg2
import select
import threading
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Function to rebuild FAISS index (Reuse your existing `build_faiss_index` function)
from rag import build_faiss_index  # Import your existing function

# Function to listen for PostgreSQL notifications
def listen_for_changes():
    conn = psycopg2.connect(
        dbname=os.getenv('DATABASE_NAME'), 
        user=os.getenv('DATABASE_USER'), 
        password=os.getenv('DATABASE_PASSWORD'), 
        host=os.getenv('DATABASE_HOST'), 
        port=os.getenv('DATABASE_PORT')
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN faiss_update;")
    
    print("Listening for database changes...")

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            continue
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            print(f"✅ Received notification: {notify.payload}")

# Run the listener in a separate thread
threading.Thread(target=listen_for_changes, daemon=True).start()

# Keep the main program running
while True:
    time.sleep(60)  # Sleep indefinitely, keeping the listener alive
