import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------------------------
# Env setup
# ---------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "todung"   # MUST match ingestion

# ---------------------------------
# Embedding model
# ---------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
)

# ---------------------------------
# Pinecone init
# ---------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------------------------------
# Query
# ---------------------------------
query = "How can Todung help in hospitals?"
query_vector = embeddings.embed_query(query)

# ---------------------------------
# Search with metadata filtering
# ---------------------------------
response = index.query(
    vector=query_vector,
    top_k=3,
    namespace=NAMESPACE,
)

# ---------------------------------
# Output
# ---------------------------------
print(f"User Query: {query}")
print("\n--- Retrieved Context ---\n")

for i, match in enumerate(response["matches"], 1):
    metadata = match["metadata"]
    content = metadata.get("_node_content") or metadata.get("page_content")

    print(f"Document {i}")
    print(f"Title: {metadata.get('title')}")
    print(content)
    print("-" * 60)
