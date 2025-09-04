import os
import re
import chromadb
from langchain_groq import ChatGroq as groq
from langchain_ollama import OllamaLLM as ollama
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI as gemini

MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded successfully.")

vector_db = chromadb.PersistentClient(path="./chroma_db")
collection_name = "elixire_docs_bge_large"
collection = vector_db.get_collection(name=collection_name)
print(f"Connected to ChromaDB collection: {collection_name}")

llm_local = ollama(model="deepseek-r1:8b")
llm_groq = groq(model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
llm_gemini = gemini(model="gemini-2.5-pro")

# --- Conversation memory ---
chat_history = []


def get_relevant_chunks(query: str, n_results: int = 1) -> list:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0] if results['documents'] else []


def generate_response(user_message: str, context_chunks: list, llm_choice: str, step_mode: bool = True) -> str:
    context = "\n\n".join(context_chunks)

    system_prompt = f"""
    You are the Elixire Assistant — a helpful chatbot inside Elixire, a pharmacy management solution.

    GOAL:
    - Guide the user through tasks step by step.
    - In each response, give ONLY the next step, prefixed with "Step X:".
    - End with a friendly reminder like: "👉 Once you complete this step, type 'done' and I’ll guide you further."

    CONTEXT USE:
    - Base your answer strictly on the provided context.
    - If the query is ambiguous, state one simple assumption before answering.
    - Never invent features or information not present in the context.
    - If the context does not contain the answer, clearly say "I don’t have enough information from Elixire to answer this."

    STYLE:
    - Professional, friendly, plain language.
    - Avoid jargon (or explain briefly if used).
    - Never provide medical, legal, or regulatory advice.

    Context:
    {context}
    """


    messages = [SystemMessage(content=system_prompt)] + chat_history + [HumanMessage(content=user_message)]

    print(f"\n--- Sending prompt to {llm_choice.upper()} ---")

    if llm_choice == "ollama" and llm_local:
        response = llm_local.invoke(messages)
    elif llm_choice == "groq" and llm_groq:
        response = llm_groq.invoke(messages).content
    elif llm_choice == "gemini" and llm_gemini:
        response = llm_gemini.invoke(messages).content
    else:
        return f"Error: The selected model '{llm_choice}' is not available."

    # --- Update conversation memory ---
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response))

    return response


def format_llm_output(response: str) -> str:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", response)
    response = re.sub(r"\n\s*\n", "\n\n", response.strip())
    return response


def main():
    llm_option = ""
    while llm_option not in ["groq", "ollama", "gemini"]:
        choice = input("Choose the LLM for inference (groq/ollama/gemini): ").lower().strip()
        if choice == "groq" and llm_groq:
            llm_option = "groq"
        elif choice == "ollama" and llm_local:
            llm_option = "ollama"
        elif choice == "gemini" and llm_gemini:
            llm_option = "gemini"
        else:
            print("Invalid choice or model not available. Please try again.")

    print("\nEnter your message. Type 'quit' or 'exit' to end the chat.")
    while True:
        user_message = input("\nYou: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # 🔹 Only fetch context if it's not just "done"
        if user_message.lower() == "done":
            relevant_chunks = []
            print("\n🔄 Continuing to the next step...\n")
        else:
            print("\n📚 Searching knowledge base for relevant context...")
            relevant_chunks = get_relevant_chunks(user_message)
            print(f"✅ Retrieved {len(relevant_chunks)} relevant chunk(s).\n")
        
        print("🤖 Generating response from LLM...")
        final_response = generate_response(user_message, relevant_chunks, llm_option, step_mode=True)
        formatted = format_llm_output(final_response)
        
        print(f"\n--- 💬 {llm_option.upper()}'s Response ---\n")
        print(formatted)
        print("\n--- End of response ---\n")



if __name__ == "__main__":
    main()
