import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "default"

# ---------------------------------
# Load text file
# ---------------------------------
def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------
# Chunk text
# ---------------------------------
def split_text(text: str, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------------------------
# Main ingestion logic
# ---------------------------------
def ingest():
    print("🔹 Loading knowledge base...")
    raw_text = load_text_file("Todung_knowledgebase.txt")

    chunks = split_text(raw_text)
    print(f"🔹 Created {len(chunks)} chunks")

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "Todung_knowledgebase.txt"}
        )
        for chunk in chunks
    ]

    print("🔹 Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT"
    )

    # ---------------------------------
    # Pinecone setup
    # ---------------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print("🔹 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,           # REQUIRED for text-embedding-004
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("🔹 Uploading embeddings to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE
    )

    print("✅ Embeddings successfully stored in Pinecone")

# ---------------------------------
# Entry point
# ---------------------------------
if __name__ == "__main__":
    ingest()
