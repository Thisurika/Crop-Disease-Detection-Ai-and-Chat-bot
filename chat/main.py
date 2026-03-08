# chat/main.py
"""
Standalone Chatbot Test Script with Conversation Memory
Remembers previous messages in the session for better context
"""

from chatbot import get_chat_response

def main():
    print("\n" + "="*60)
    print("🌿 Plant AI Chatbot - Test Mode (with Memory)")
    print("Ask anything about plant diseases, treatment, prevention...")
    print("Type 'exit', 'quit', or 'clear' to stop or reset history")
    print("="*60 + "\n")

    # Conversation history (list of dicts: {"role": "user/assistant", "content": "..."})
    history = []

    while True:
        query = input("You: ").strip()

        if query.lower() in ['exit', 'quit', 'q', 'bye']:
            print("\nGoodbye! Stay green 🌱")
            break

        if query.lower() == 'clear':
            history = []
            print("\nChat history cleared.\n")
            continue

        if not query:
            print("Please type something...\n")
            continue

        disease = input("Detected disease (optional, press Enter to skip): ").strip() or None

        print("\nAI thinking...")

        try:
            # Build full context for Groq (history + current query + disease)
            messages = history.copy()  # copy to avoid modifying original

            if disease:
                messages.append({
                    "role": "system",
                    "content": f"Current detected disease: {disease}. Tailor advice to this."
                })

            messages.append({"role": "user", "content": query})

            # Call chatbot with full context
            response = get_chat_response(query, disease=disease)

            # Show response
            print(f"AI: {response}\n")

            # Save to history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please check internet or Groq API key in .env\n")

if __name__ == "__main__":
    main()