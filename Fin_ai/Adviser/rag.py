import os
import psycopg2
import json
from together import Together
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Together client
API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=API_KEY)


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

# Function to fetch financial data and build context
def fetch_financial_data():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Fetch category data
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

        # Merge all data into readable context
        all_data = []
        for category, details in category_data_dict.items():
            transactions_text = " | ".join(details["transactions"]) if details["transactions"] else "No transactions"
            expenses_text = " | ".join(details["expenses"]) if details["expenses"] else "No expenses"
            budget_text = details["budget"]

            # Calculate total amounts
            total_transactions_amount = sum(float(tx.split("Transactions_Amount: ")[1]) for tx in details["transactions"] if "Transactions_Amount: " in tx)
            total_expenses_amount = sum(float(exp.split("Expense_Amount: ")[1]) for exp in details["expenses"] if "Expense_Amount: " in exp)
            total_amount=total_transactions_amount+total_expenses_amount
            final_text = f"Category: {category}, {budget_text}, Transactions: {transactions_text}, Expenses: {expenses_text}, Total Amount: {total_amount}"
            all_data.append(final_text)

        return all_data

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

# Generate RAG response using all_data directly as context
def get_rag_response(query):
    # Get financial data as context
    all_data = fetch_financial_data()
    context_text = "\n".join(all_data)

    # Construct the prompt
    prompt = f"""
    You are a financial AI assistant. Based on the given financial data, analyze and respond accordingly.

    Context:
    {context_text}

    Question:
    {query}

    Answer:
    """

    return query_llama3(prompt)


