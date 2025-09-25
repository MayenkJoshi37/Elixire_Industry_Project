import os
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq as groq


# ----------------------------
# Load Embedding Model
# ----------------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"[Info] Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("[Info] Embedding model loaded successfully.")

# ----------------------------
# Connect to ChromaDB
# ----------------------------
vector_db = chromadb.PersistentClient(path="./chroma_db")
collection_name = "elixire_docs_bge_large"
collection = vector_db.get_collection(name=collection_name)
print(f"[Info] Connected to ChromaDB collection: {collection_name}")

# ----------------------------
# Initialize Groq LLM
# ----------------------------
llm_groq = groq(model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))


# ----------------------------
# Helper Functions
# ----------------------------
def get_relevant_chunks(query: str, n_results: int = 1) -> list:
    """Retrieve context chunks from ChromaDB using embeddings."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


def generate_response(user_message: str, context_chunks: list) -> str:
    """Generate English response from Groq LLM given context."""
    context = "\n\n".join(context_chunks)
    system_prompt = f"""
    You are the Elixire Assistant — a helpful chatbot inside Elixire, a pharmacy management solution.
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

    print(f"\n--- Sending prompt to GROQ ---")
    return llm_groq.invoke(messages).content


def format_llm_output(response: str) -> str:
    """Clean and format LLM output for terminal display."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", response)
    response = re.sub(r"\n\s*\n", "\n\n", response.strip())
    return response


def save_prompt_to_file(system_prompt: str, user_message: str, folder_name="llm_prompts"):
    """Save system + user prompt for debugging."""
    os.makedirs(folder_name, exist_ok=True)

    import time
    timestamp = int(time.time())
    file_path = os.path.join(folder_name, f"prompt_{timestamp}.txt")

    full_prompt = f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n--- USER MESSAGE ---\n{user_message}\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    print(f"[Info] Prompt saved to {file_path}")


# ----------------------------
# Pre- and Post-processing with Groq
# ----------------------------
def preprocess_user_query(user_message: str) -> dict:
    """
    Step 1: Use Groq to refine English query or translate non-English query into English.
    Returns JSON with user_query_eng and user_original_query_lang.
    """
    system_prompt = """
    You are a query pre-processor for a multilingual assistant.
    - If the user query is in English: refine/clean it.
    - If the user query is not in English: translate it into clear English.
    - Always detect the user's original language (use ISO 639-1 code if possible, else language name).
    - Respond ONLY in valid JSON, no extra text.
    Format:
    {
      "user_query_eng": "<refined_or_translated_query_in_english>",
      "user_original_query_lang": "<detected_language_code_or_name>"
    }
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    raw_response = llm_groq.invoke(messages).content

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("[Warning] Groq did not return valid JSON, falling back.")
        return {
            "user_query_eng": user_message,
            "user_original_query_lang": "en",
        }


def normalize_lang_code(lang: str) -> str:
    """Normalize language code/name to a consistent form."""
    lang = lang.strip().lower()
    mapping = {
        "en": "English",
        "english": "English",
        "hi": "Hindi",
        "hindi": "Hindi",
        "mr": "Marathi",
        "marathi": "Marathi",
        "bn": "Bengali",
        "bengali": "Bengali",
        "gu": "Gujarati",
        "gujarati": "Gujarati",
    }
    return mapping.get(lang, lang.capitalize())


def postprocess_answer(answer_eng: str, target_lang: str) -> str:
    """
    Step 3: Translate the English answer back into the user's original language (if not English).
    """
    target_lang = normalize_lang_code(target_lang)

    if target_lang == "English":
        return answer_eng

    system_prompt = f"""
    You are a translator.
    Convert the following English text into {target_lang}.
    - Keep the same numbering, steps, and structure.
    - Do not add extra commentary.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=answer_eng),
    ]

    try:
        return llm_groq.invoke(messages).content
    except Exception as e:
        print(f"[Warning] Translation failed: {e}")
        return answer_eng


# ----------------------------
# Main Chat Loop
# ----------------------------
def main():
    print("\n[Info] Running in GROQ-only mode.")
    print("Enter your message. Type 'quit' or 'exit' to end the chat.")

    while True:
        user_message = input("\nYou: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Step 1: preprocess user query
        print("[Step 1] Preprocessing user query...")
        query_data = preprocess_user_query(user_message)
        user_query_eng = query_data["user_query_eng"]
        user_lang = query_data["user_original_query_lang"]
        print(f"[Info] Refined/translated query: {user_query_eng}")
        print(f"[Info] Original language: {user_lang}")

        # Step 2: retrieve context
        print("[Step 2] Searching knowledge base for relevant context...")
        relevant_chunks = get_relevant_chunks(user_query_eng)
        print(f"[Info] Retrieved {len(relevant_chunks)} relevant chunk(s).")

        # Step 3: generate English answer
        print("[Step 3] Generating response from Groq...")
        final_response_eng = generate_response(user_query_eng, relevant_chunks)

        # Step 4: postprocess answer into original language (if needed)
        print("[Step 4] Translating answer back (if required)...")
        final_response = postprocess_answer(final_response_eng, user_lang)

        formatted = format_llm_output(final_response)
        print(f"\n--- GROQ's Response ---\n")
        print(formatted)
        print("\n--- End of response ---\n")


if __name__ == "__main__":
    main()
