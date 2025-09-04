import os
import re
import chromadb
from langchain_groq import ChatGroq as groq
from langchain_ollama import OllamaLLM as ollama
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
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

def get_relevant_chunks(query: str, n_results: int = 1) -> list:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0] if results['documents'] else []

def generate_response(user_message: str, context_chunks: list, llm_choice: str) -> str:
    context = "\n\n".join(context_chunks)
    system_prompt = f"""
    You are the Elixire Assistant — a helpful chatbot inside Elixire, A pharmacy management solution.
    Your job is to give users a simple, concise and clear answer.
    Follow these rules for answering:

    GOAL:
    - Give clear, accurate, and concise answers for non-technical users.
    - Entire response must be short and skimmable

    STRUCTURE:
    1. Provide numbered, actionable steps.
    2. Add up to 1 troubleshooting tip only if critical.

    CONTEXT USE:
    - Base your answer strictly on the provided context.
    - If the query is ambiguous, state one simple assumption before answering.
    - Never invent features or information not present in the context.

    STYLE:
    - Professional, friendly, plain language.
    - Avoid jargon (or explain briefly if used).
    - Never provide medical, legal, or regulatory advice.

    Context:
    {context}
    """

    save_prompt_to_file(system_prompt, user_message)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    print(f"\n--- Sending prompt to {llm_choice.upper()} ---")

    if llm_choice == "ollama" and llm_local:
        return llm_local.invoke(messages)
    elif llm_choice == "groq" and llm_groq:
        return llm_groq.invoke(messages).content
    elif llm_choice == "gemini" and llm_gemini:
        return llm_gemini.invoke(messages).content
    else:
        return f"Error: The selected model '{llm_choice}' is not available."

def format_llm_output(response: str) -> str:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", response)
    response = re.sub(r"\n\s*\n", "\n\n", response.strip())
    return response

def save_prompt_to_file(system_prompt: str, user_message: str, folder_name="llm_prompts"):
    os.makedirs(folder_name, exist_ok=True)
    
    import time
    timestamp = int(time.time())
    file_path = os.path.join(folder_name, f"prompt_{timestamp}.txt")
    
    full_prompt = f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n--- USER MESSAGE ---\n{user_message}\n"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)
    
    print(f"[Info] Prompt saved to {file_path}")


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

        print("Searching knowledge base for relevant context...")
        relevant_chunks = get_relevant_chunks(user_message)
        print(f"Retrieved {len(relevant_chunks)} relevant chunk(s).")
        print("Generating response from LLM...")
        final_response = generate_response(user_message, relevant_chunks, llm_option)
        formatted = format_llm_output(final_response)
        print(f"\n--- {llm_option.upper()}'s Response ---\n")
        print(formatted)
        print("\n--- End of response ---\n")

if __name__ == "__main__":
    main()