import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -------------------------------
# Load ONE TXT document
# -------------------------------
def load_single_txt_file(txt_file_path):
    print(f"Loading document from {txt_file_path}...")

    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"{txt_file_path} does not exist")

    with open(txt_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Document loaded successfully")
    print(f"Content length: {len(text)} characters")

    return text


# -------------------------------
# Manual text chunking
# -------------------------------
def split_text(text, chunk_size=500, chunk_overlap=50):
    print("Splitting text into chunks...")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    print(f"Created {len(chunks)} chunks")
    return chunks


# -------------------------------
# Create Pinecone Vector Store
# -------------------------------
def create_vector_store(
    chunks,
    index_name="chatbot-index",
    dimension=384,
    namespace="default"
):
    print("Creating embeddings and storing in Pinecone...")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    pc = Pinecone(api_key="pcsk_4vS2VH_3Z5Ck19AqgWSNebcJSyCpaRsVBddS4BrW1shwwzj65VLPyivimbLsD261utLoft")

    if index_name not in pc.list_indexes().names():
        print("Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    vectors = []
    embeddings = embedder.encode(chunks)

    for i, embedding in enumerate(embeddings):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunks[i],
                "source": "Todung_knowledgebase.txt"
            }}
        )

    index.upsert(vectors=vectors, namespace=namespace)

    print("Pinecone vector store created successfully")


# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    TXT_FILE_PATH = "Todung_knowledgebase.txt"

    text = load_single_txt_file(TXT_FILE_PATH)
    chunks = split_text(text)
    create_vector_store(chunks)

    print("\n✅ Ingestion complete! Your data is now ready for RAG.")


if __name__ == "__main__":
    main()
