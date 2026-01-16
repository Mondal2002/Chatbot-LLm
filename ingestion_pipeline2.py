import os
import re
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

INDEX_NAME = "gemini-rag-index3"
NAMESPACE = "todung"

# ---------------------------------
# Load text file
# ---------------------------------
def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------
# Parse embedding-ready chunks
# ---------------------------------
def parse_chunks(text: str):
    raw_chunks = re.split(r"=== CHUNK \d+ ===", text)
    documents = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        title = re.search(r"TITLE:\s*(.*)", chunk)
        section = re.search(r"SECTION:\s*(.*)", chunk)
        industry = re.search(r"INDUSTRY:\s*(.*)", chunk)

        content = re.sub(
            r"TITLE:.*\nSECTION:.*\nINDUSTRY:.*\n",
            "",
            chunk,
            flags=re.DOTALL
        ).strip()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "title": title.group(1).strip() if title else "unknown",
                    "section": section.group(1).strip() if section else "unknown",
                    "industry": industry.group(1).strip() if industry else "generic",
                    "source": "Todung_knowledgebase2.txt"
                }
            )
        )

    return documents

# ---------------------------------
# Main ingestion logic
# ---------------------------------
def ingest():
    print("🔹 Loading knowledge base...")
    raw_text = load_text_file("Todung_knowledgebase2.txt")

    documents = parse_chunks(raw_text)
    print(f"🔹 Parsed {len(documents)} semantic chunks")

    print("🔹 Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print("🔹 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
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

    print("✅ Todung knowledge base successfully indexed")

# ---------------------------------
# Entry point
# ---------------------------------
if __name__ == "__main__":
    ingest()
