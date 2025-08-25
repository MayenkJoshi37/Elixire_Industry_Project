from sentence_transformers import SentenceTransformer
import chromadb

# --- SETUP ---

# 1. Embedding Model: BGE-Large
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")

# 2. ChromaDB Client
vector_db = chromadb.PersistentClient(path="./chroma_db")

# 3. ChromaDB Collection Setup
collection_name = "elixire_docs_bge_large"
collection = vector_db.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Cosine similarity is effective for these embeddings
)

# --- FUNCTIONS ---

def chunk_text_by_paragraph(text: str) -> list[str]:
    """
    Splits text into non-empty paragraphs.
    """
    return [p.strip() for p in text.split('\n') if p.strip()]

def add_documents_to_db(text_chunks: list[str], source_id: str) -> None:
    """
    Encodes text chunks and adds them to the ChromaDB collection.
    """
    print(f"Encoding {len(text_chunks)} text chunks...")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True).tolist()
    print("Encoding complete.")

    ids = [f"{source_id}_{i}" for i in range(len(text_chunks))]

    collection.add(
        embeddings=embeddings,
        documents=text_chunks,
        ids=ids
    )
    print(f"Added {len(text_chunks)} documents to the '{collection_name}' collection.")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    file_path = 'text_cleaned.txt'
    print(f"Reading and processing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        chunks = chunk_text_by_paragraph(text)
        add_documents_to_db(chunks, "elixire_doc")

        print("\n--- Success! ---")
        print(f"Vectorization complete. {len(chunks)} chunks added.")
        print(f"Total documents in collection: {collection.count()}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
