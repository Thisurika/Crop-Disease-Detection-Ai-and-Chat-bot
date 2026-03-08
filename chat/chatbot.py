# chat/chatbot.py
"""
Plant Disease & Crop Advice Chatbot
- Groq LLM (current model) + FAQ CSV fallback
- Tailored for Sri Lanka farmers
"""

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

client = Groq(api_key=GROQ_API_KEY)

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

    # Step 2: Groq LLM
    try:
        system_prompt = """
You are an expert agricultural advisor in Sri Lanka, helping farmers with plant diseases, pests, and crop care.
- Use very simple, practical Sinhala/English mixed language if possible (keep it easy).
- Suggest affordable, natural, locally available solutions first (neem oil, cow urine, ash, turmeric, etc.).
- Recommend chemical options only as last resort with safety warnings.
- Focus on prevention, monsoon season tips, and Negombo/Western Province conditions.
- Be encouraging, patient, and helpful.
- Keep answers short and clear (max 150-200 words).
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add history if provided (for future memory support)
        if history:
            messages.extend(history)

        # Add disease context
        user_message = f"Question: {query}"
        if disease:
            user_message += f"\nDetected disease: {disease}"

        messages.append({"role": "user", "content": user_message})

        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",          # Current, supported, high-quality model
            temperature=0.7,
            max_tokens=400,
            top_p=0.9
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)
        if "decommissioned" in error_msg or "model" in error_msg:
            return "Sorry, the AI model is temporarily unavailable. Please try again later or contact a local agriculture officer."
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