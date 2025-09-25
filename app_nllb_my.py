import os
import re
import time
import torch
import chromadb
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq as groq
from langchain_ollama import OllamaLLM as ollama
from langchain_google_genai import ChatGoogleGenerativeAI as gemini

# ------------------ Embedding Setup ------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"[Info] Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("[Info] Embedding model loaded successfully.")

vector_db = chromadb.PersistentClient(path="./chroma_db")
collection_name = "elixire_docs_bge_large"
collection = vector_db.get_collection(name=collection_name)
print(f"[Info] Connected to ChromaDB collection: {collection_name}")

# ------------------ LLM Clients ------------------
llm_local = ollama(model="deepseek-r1:8b")  # adjust as needed
llm_groq = groq(model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
llm_gemini = gemini(model="gemini-2.5-pro")

# ------------------ NLLB Translation Setup ------------------
DetectorFactory.seed = 0
NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"  # good quality, fits 8GB GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] Loading NLLB model: {NLLB_MODEL} on {device} ...")
tokenizer_nllb = AutoTokenizer.from_pretrained(NLLB_MODEL)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device)
print("[Info] NLLB model loaded successfully.")

# Supported languages
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr"
}

# ------------------ Language Detection ------------------
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang in NLLB_LANG_MAP:
            return lang
        return "en"
    except Exception:
        return "en"

def translate_text(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text

    # Map your languages to NLLB codes
    src_tag = NLLB_LANG_MAP.get(src, "eng_Latn")
    tgt_tag = NLLB_LANG_MAP.get(tgt, "eng_Latn")

    # Ensure the target tag exists in the tokenizer
    if tgt_tag not in tokenizer_nllb.lang_code_to_id:
        print(f"[Warning] Target language {tgt_tag} not in tokenizer. Falling back to English.")
        tgt_tag = "eng_Latn"

    tokenizer_nllb.src_lang = src_tag
    inputs = tokenizer_nllb(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    forced_bos_token_id = tokenizer_nllb.lang_code_to_id.get(tgt_tag, None)

    gen_kwargs = {"max_length": 512, "num_beams": 4}
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        translated_ids = model_nllb.generate(**inputs, **gen_kwargs)

    return tokenizer_nllb.batch_decode(translated_ids, skip_special_tokens=True)[0]

# ------------------ Core Functions ------------------
def get_relevant_chunks(query: str, n_results: int = 1) -> list:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0] if results['documents'] else []

def generate_response(user_message: str, context_chunks: list, llm_choice: str) -> str:
    context = "\n\n".join(context_chunks)
    system_prompt = f"""
You are the Elixire Assistant — a helpful chatbot inside Elixire, a pharmacy management solution.
Your job is to give users a simple, concise and clear answer.

GOAL:
- Give clear, accurate, concise answers for non-technical users.
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

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    print(f"[Info] Sending prompt to {llm_choice.upper()}...")

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
    timestamp = int(time.time())
    file_path = os.path.join(folder_name, f"prompt_{timestamp}.txt")
    full_prompt = f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n--- USER MESSAGE ---\n{user_message}\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)
    print(f"[Info] Prompt saved to {file_path}")

# ------------------ Main Loop ------------------
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

        # Detect language
        user_lang = detect_language(user_message)
        print(f"[Info] Detected language: {user_lang}")

        # Translate query → English for embeddings & LLM
        query_for_llm = user_message
        if user_lang != "en":
            print("[Info] Translating query → English...")
            query_for_llm = translate_text(user_message, src=user_lang, tgt="en")

        # Retrieve context
        print("[Info] Searching knowledge base...")
        relevant_chunks = get_relevant_chunks(query_for_llm)
        print(f"[Info] Retrieved {len(relevant_chunks)} relevant chunk(s).")

        # Generate response in English
        response = generate_response(query_for_llm, relevant_chunks, llm_option)
        if isinstance(response, dict) and "content" in response:
            final_text = response["content"]
        else:
            final_text = response

        # Translate back → user language
        if user_lang != "en":
            print(f"[Info] Translating response → {user_lang}...")
            final_text = translate_text(final_text, src="en", tgt=user_lang)

        # Format and print
        formatted = format_llm_output(final_text)
        print(f"\n--- {llm_option.upper()}'s Response ---\n")
        print(formatted)
        print("\n--- End of response ---\n")

if __name__ == "__main__":
    main()
