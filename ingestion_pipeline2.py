import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

import boto3
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "titan-rag-index"
NAMESPACE = "default"
EMBEDDING_DIM = 1024

# ---------------------------------
# Bedrock client
# ---------------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# ---------------------------------
# Custom Titan Embeddings Wrapper
# ---------------------------------
class TitanEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        body = {
            "inputText": text,
            "dimensions": EMBEDDING_DIM
        }

        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

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
    raw_text = load_text_file("Todung_knowledgebase2.txt")

    chunks = split_text(raw_text)
    print(f"🔹 Created {len(chunks)} chunks")

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "Todung_knowledgebase2.txt"}
        )
        for chunk in chunks
    ]

    print("🔹 Initializing Titan embeddings...")
    embeddings = TitanEmbeddings()

    # ---------------------------------
    # Pinecone setup
    # ---------------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print("🔹 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
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
