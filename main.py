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

def get_relevant_chunks(query: str, n_results: int = 2) -> list:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0] if results['documents'] else []

def generate_response(user_message: str, context_chunks: list, llm_choice: str) -> str:
    context = "\n\n".join(context_chunks)
    system_prompt = f"""
    You are a helpful assistant. Use the following context to answer the user question.
    Provide a clear Step-by-Step answer.

    Context:
    {context}
    """
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




"""

1. Proper sys prompt (n-shot prompting, explanation all that)
2. make formating general
3. if possible add a simple front end


git push
"""