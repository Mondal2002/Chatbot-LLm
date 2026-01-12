import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -------------------------------
# Environment variables
# -------------------------------
PINECONE_API_KEY = "pcsk_4vS2VH_3Z5Ck19AqgWSNebcJSyCpaRsVBddS4BrW1shwwzj65VLPyivimbLsD261utLoft"
INDEX_NAME = "chatbot-index"
NAMESPACE = "default"  # must match ingestion

# -------------------------------
# Sentence Transformers embeddings
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Pinecone initialization
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

db = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    namespace=NAMESPACE
)

# -------------------------------
# Retriever
# -------------------------------
query = "How can Todung be useful to hospitals"

retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# -------------------------------
# Prompt
# -------------------------------
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't have enough information to answer that question."

Context:
{context}

Question:
{question}

Answer:
"""
)

context_text = "\n".join(doc.page_content for doc in relevant_docs)

# -------------------------------
# LLaMA via Ollama (FREE)
# -------------------------------
llm = OllamaLLM(model="tinyllama")

chain = (
    prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({
    "context": context_text,
    "question": query
})

print("\n--- Generated Response ---")
print(result)
