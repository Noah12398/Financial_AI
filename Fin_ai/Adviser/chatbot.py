import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_expenses(transactions):
    categories = {}
    for t in transactions:
        categories[t.category] = categories.get(t.category, 0) + t.amount

    summary = "\n".join([f"{cat}: ${amt}" for cat, amt in categories.items()])
    prompt = f"My expenses: \n{summary}\nWhat are my unnecessary expenses?"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response["choices"][0]["message"]["content"]
