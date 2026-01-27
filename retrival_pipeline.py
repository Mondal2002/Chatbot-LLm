import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "default"

# Load model
model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
)

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Query
query = "How can Todung help in hospitals?"

# Embed query
# FIX: Use .embed_query() instead of .encode()
query_vector = model.embed_query(query)

# Search
response = index.query(
    vector=query_vector,
    top_k=1,
    include_metadata=True,
    namespace=NAMESPACE
)

print(f"User Query: {query}")
print("--- Context ---")

for i, match in enumerate(response["matches"], 1):
    print(f"Document {i}:")
    print(match["metadata"]["text"])
    print("-" * 50)
