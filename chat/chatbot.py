# chat/chatbot.py
"""
Plant Disease & Crop Advice Chatbot
- Groq LLM (current model) + FAQ CSV fallback
- Tailored for Sri Lanka farmers
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# FAQ file
FAQ_PATH = "plant_faq.csv"

if os.path.exists(FAQ_PATH):
    faq_df = pd.read_csv(FAQ_PATH)
    print(f"Loaded {len(faq_df)} FAQ entries")
else:
    faq_df = pd.DataFrame(columns=["disease", "advice"])
    print(f"FAQ file not found: {FAQ_PATH}. Using LLM only.")

# ────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────
def get_chat_response(query: str, disease: str = None, history: list = None) -> str:
    """
    Get AI response
    - First check FAQ for exact disease match
    - Then use Groq LLM with context (disease + optional history)
    """
    # Step 1: FAQ exact match
    if disease:
        disease_norm = disease.lower().replace(" ", "_").replace("-", "_")
        match = faq_df[faq_df['disease'].str.lower().str.contains(disease_norm, na=False)]
        if not match.empty:
            return match.iloc[0]['advice']

    # Step 2: Groq LLM (via REST API)
    try:
        if not GROQ_API_KEY:
            return "Setup required: Please add your GROQ_API_KEY to the .env file to enable the AI Chatbot."

        system_prompt = """
You are an expert agricultural advisor in Sri Lanka, helping farmers with plant diseases, pests, and crop care.
- Use very simple, practical Sinhala/English mixed language if possible (keep it easy).
- Suggest affordable, natural, locally available solutions first (neem oil, cow urine, ash, turmeric, etc.).
- Recommend chemical options only as last resort with safety warnings.
- Focus on prevention, monsoon season tips, and Negombo/Western Province conditions.
- Be encouraging, patient, and helpful.
- Keep answers short and clear (max 150-200 words).
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})

        user_message = f"Question: {query}"
        if disease:
            user_message += f"\nDetected disease: {disease}"

        messages.append({"role": "user", "content": user_message})

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 400
        }

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        if response.status_code != 200:
            return f"Sorry, AI model error: {response_data.get('error', {}).get('message', 'Unknown error')}"

        return response_data['choices'][0]['message']['content'].strip()

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "authentication" in error_msg.lower():
            return "Sorry, the AI model is temporarily unavailable due to invalid API Key credentials. Please try again later."
        return f"Sorry, I couldn't connect right now (error: {error_msg}). Try again or ask a local expert."


# ────────────────────────────────────────────────
# Standalone test mode
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Plant AI Chatbot Test Mode (with memory support) ===")
    print("Type your question. Type 'exit' to quit, 'clear' to reset history.\n")

    history = []  # Stores conversation for context

    while True:
        query = input("You: ").strip()

        if query.lower() in ['exit', 'quit', 'q', 'bye']:
            print("\nGoodbye! Protect your crops 🌱")
            break

        if query.lower() == 'clear':
            history = []
            print("\nChat history cleared.\n")
            continue

        if not query:
            print("Please type a question...\n")
            continue

        disease = input("Detected disease (optional, press Enter to skip): ").strip() or None

        print("\nAI thinking...")

        try:
            response = get_chat_response(query, disease, history=history)
            print(f"AI: {response}\n")

            # Update history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

            # Optional: limit history length to avoid token limits
            if len(history) > 12:  # ~6 turns
                history = history[-12:]

        except Exception as e:
            print(f"Error: {str(e)}\n")
