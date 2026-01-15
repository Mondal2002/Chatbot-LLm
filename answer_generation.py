import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)


load_dotenv()

# -------------------------------
# Environment variables
# -------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "default" # must match ingestion

# -------------------------------
# Sentence Transformers embeddings
# -------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
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
query = "How can Todung be useful to hotels?"

retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# -------------------------------
# Prompt
# -------------------------------
# prompt = PromptTemplate.from_template(
#     """
# Your name is Todung ,You are an helpful assistant.
# rules:
# Rules:
# - Answer in ONE short sentence.
# - Use ONLY the provided context.
# - If the answer is not present, say: I don't have enough information.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

context_text = "\n".join(doc.page_content for doc in relevant_docs)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

final_prompt = f"""
You are Todung, a helpful assistant.
for normal greetings greet them back like if someone says hi, reply:hello there, how can i help you?
Rules:
- Answer in ONE short sentence.
- Use ONLY the provided context.
- If the answer is not present, say: I don't have enough information.

Context:
{context_text}

User Question:
{query}
"""

response = llm.invoke(final_prompt)
answer = response.content.strip()

print("\n--- Generated Response ---")
print(answer)


# -------------------------------
# LLaMA via Ollama (FREE)
# -------------------------------
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2,
# )

# chain = (
#     prompt
#     | llm
#     | StrOutputParser()
# )

# result = chain.invoke({
#     "context": context_text,
#     "question": query
# })

# print("\n--- Generated Response ---")
# print(result)
