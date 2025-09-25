import os
import re
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq as groq
from langchain_ollama import OllamaLLM as ollama
from langchain_google_genai import ChatGoogleGenerativeAI as gemini

# ---------------- Embedding Model ----------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded successfully.")

vector_db = chromadb.PersistentClient(path="./chroma_db")
collection_name = "elixire_docs_bge_large"
collection = vector_db.get_collection(name=collection_name)
print(f"Connected to ChromaDB collection: {collection_name}")

# ---------------- LLM Clients ----------------
llm_local = ollama(model="ollama run llama3.2:3b")
llm_groq = groq(model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))


# ---------------- NLLB Translation Setup ----------------
DetectorFactory.seed = 0
NLLB_MODEL = "facebook/nllb-200-distilled-600M"   # use 3.3B on GPU if available
print(f"Loading NLLB model: {NLLB_MODEL} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_nllb = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn", tgt_lang="eng_Latn")
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device)
print("NLLB loaded.")

NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
}

def detect_language(text: str) -> str:
    try:
        ld = detect(text)
        if ld.startswith("hi"):
            return "hi"
        if ld.startswith("mr"):
            return "mr"
        if ld.startswith("en"):
            return "en"
        return "en"
    except Exception:
        return "en"

def translate_text(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    src_tag = NLLB_LANG_MAP.get(src, "eng_Latn")
    tgt_tag = NLLB_LANG_MAP.get(tgt, "eng_Latn")

    tokenizer_nllb.src_lang = src_tag
    inputs = tokenizer_nllb(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    forced_bos_token_id = None
    if hasattr(tokenizer_nllb, "lang_code_to_id") and tgt_tag in tokenizer_nllb.lang_code_to_id:
        forced_bos_token_id = tokenizer_nllb.lang_code_to_id[tgt_tag]

    gen_kwargs = {"max_length": 512}
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        translated_ids = model_nllb.generate(**inputs, **gen_kwargs)

    return tokenizer_nllb.batch_decode(translated_ids, skip_special_tokens=True)[0]

# ---------------- Core Functions ----------------
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

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
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
    response = re.sub(r"\\(.?)\\*", r"\033[1m\1\033[0m", response)
    response = re.sub(r"\n\s*\n", "\n\n", response.strip())
    return response

# ---------------- Main Loop ----------------
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

        user_lang = detect_language(user_message)
        print(f"[lang] Detected: {user_lang}")

        # Translate query to English for retrieval
        query_for_retrieval = user_message
        if user_lang != "en":
            print("[translating] → English")
            query_for_retrieval = translate_text(user_message, src=user_lang, tgt="en")

        print("Searching knowledge base for relevant context...")
        relevant_chunks = get_relevant_chunks(query_for_retrieval)
        print(f"Retrieved {len(relevant_chunks)} relevant chunk(s).")

        print("Generating response from LLM...")
        response = generate_response(query_for_retrieval, relevant_chunks, llm_option)

        if isinstance(response, dict) and "content" in response:
            final_text = response["content"]
        else:
            final_text = response

        # Translate back to user language if needed
        if user_lang != "en":
            print(f"[translating] English → {user_lang}")
            final_text = translate_text(final_text, src="en", tgt=user_lang)

        formatted = format_llm_output(final_text)
        print(f"\n--- {llm_option.upper()}'s Response ---\n")
        print(formatted)
        print("\n--- End of response ---\n")

if _name_ == "_main_":
    main()