import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = "pcsk_4vS2VH_3Z5Ck19AqgWSNebcJSyCpaRsVBddS4BrW1shwwzj65VLPyivimbLsD261utLoft"
INDEX_NAME = "chatbot-index"
NAMESPACE = "default"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Query
query = "How can Todung help in hospitals?"

# Embed query
query_vector = model.encode(query).tolist()

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
