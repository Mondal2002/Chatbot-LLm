import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_aws import BedrockEmbeddings

# ---------------------------------
# Environment
# ---------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "titan-rag-index"
NAMESPACE = "default"

# ---------------------------------
# Titan Embeddings (Bedrock)
# ---------------------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)

# ---------------------------------
# Pinecone
# ---------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------------------------------
# Query
# ---------------------------------
query = "How can Todung help in hospitals?"

# Embed query
query_vector = embeddings.embed_query(query)

# Safety check
assert len(query_vector) == 1024, "Embedding dimension mismatch!"

# Search
response = index.query(
    vector=query_vector,
    top_k=1,
    include_metadata=True,
    namespace=NAMESPACE,
)

print(f"User Query: {query}")
print("--- Context ---")

for i, match in enumerate(response.get("matches", []), 1):
    metadata = match.get("metadata", {})
    text = metadata.get("text") or metadata.get("page_content", "[No text found]")

    print(f"Document {i}:")
    print(text)
    print("-" * 50)
