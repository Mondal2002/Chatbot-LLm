# import os
# from dotenv import load_dotenv

# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     GoogleGenerativeAIEmbeddings
# )

# load_dotenv()

# # -------------------------------
# # Environment
# # -------------------------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "gemini-rag-index2"
# NAMESPACE = "default" # must match ingestion

# # -------------------------------
# # Embeddings (Sentence Transformers)
# # -------------------------------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=GOOGLE_API_KEY,
#     task_type="RETRIEVAL_QUERY"
# )

# # -------------------------------
# # Pinecone
# # -------------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# db = PineconeVectorStore(
#     index=index,
#     embedding=embeddings,
#     namespace=NAMESPACE
# )

# # -------------------------------
# # LLaMA via Ollama
# # -------------------------------
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2,
# )

# # -------------------------------
# # Conversation memory
# # -------------------------------
# chat_history = []

# # -------------------------------
# # Ask question function
# # -------------------------------
# def ask_question(user_question):
#     print(f"\n--- You asked: {user_question} ---")

#     # Step 1: Rewrite question using 
#     if chat_history:
#         history_text = ""
#         for msg in chat_history:
#             history_text += f"- {msg.content}\n"

#         rewrite_prompt = f"""
# Given the chat history below, rewrite the new question so that it is standalone and searchable.
# Return ONLY the rewritten question.

# Chat history:
# {history_text}

# New question:
# {user_question}
# """

#         search_question = llm.invoke(rewrite_prompt).strip()
#         print(f"Searching for: {search_question}")
#     else:
#         search_question = user_question

#     # Step 2: Retrieve documents
#     retriever = db.as_retriever(search_kwargs={"k": 3})
#     docs = retriever.invoke(search_question)

#     print(f"Found {len(docs)} relevant documents:")
#     for i, doc in enumerate(docs, 1):
#         preview_lines = doc.page_content.split("\n")[:2]
#         preview = "\n".join(preview_lines)
#         print(f"  Doc {i}: {preview}...")

#     # Step 3: Build final context (SAFE)
#     context = ""
#     for doc in docs:
#         context += f"- {doc.page_content}\n"

#     final_prompt = f"""
# You are a helpful assistant that answers questions ONLY using the provided documents.
# If the answer is not present, say:
# "I don't have enough information to answer that question based on the provided documents."

# Documents:
# {context}

# Question:
# {user_question}

# Answer:
# """

#     answer = llm.invoke(final_prompt).strip()

#     # Step 4: Save conversation
#     chat_history.append(HumanMessage(content=user_question))
#     chat_history.append(AIMessage(content=answer))

#     print(f"Answer: {answer}")
#     return answer

# # -------------------------------
# # Chat loop
# # -------------------------------





import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "default"

# ---------------------------------
# Gemini Embeddings & LLM
# ---------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Updated to current stable version
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# ---------------------------------
# Pinecone Setup
# ---------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------
# Memory Management State
# ---------------------------------
# In production, use a dictionary keyed by session_id
memory_store = {
    "conversation_summary": "",
    "recent_messages": [], # List of BaseMessages
}
MAX_RECENT_MESSAGES = 8  # N turns (3 Human + 3 AI)

def summarize_messages(summary: str, recent_msgs: list) -> str:
    """Summarizes recent messages and merges them with the existing summary."""
    history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_msgs])
    
    summary_prompt = (
        f"Current Summary: {summary}\n\n"
        f"New Messages to incorporate:\n{history_text}\n\n"
        "Generate a concise updated summary of the conversation so far."
    )
    response = llm.invoke(summary_prompt)
    return response.content.strip()

# ---------------------------------
# RAG Logic
# ---------------------------------
def ask_question(user_question: str) -> str:
    global memory_store

    # 1. Load session memory
    summary = memory_store["conversation_summary"]
    recent_msgs = memory_store["recent_messages"]

    # 2. Build History Text for Query Rewriting/Context
    history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_msgs])
    
    # 3. Rewrite question to be standalone (Context-Aware)
    rewrite_prompt = (
        f"Summary of conversation: {summary}\n"
        f"Recent History: {history_text}\n"
        f"Question: {user_question}\n"
        "Given the history and summary, rewrite the question to be a standalone search query."
    )
    search_question = llm.invoke(rewrite_prompt).content.strip()

    # 4. Retrieve context from Pinecone
    docs = retriever.invoke(search_question)
    context = "\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."

    # 5. Build final prompt with Summary + Recent + Context
    final_prompt = f"""
You are Todung, a helpful assistant.
Rules:
- For greetings (hi, hello, etc.), reply: "Hello, I am Todung. How can I help you?"
- Answer in ONE short sentence.
- Use the provided Context and Conversation Summary to inform your answer.
- If the answer is not in the context, say: I don't have enough information.

Conversation Summary: {summary}
Recent History: {history_text}
Context: {context}

User Question: {user_question}
"""

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # 6. Update recent_messages
    recent_msgs.append(HumanMessage(content=user_question))
    recent_msgs.append(AIMessage(content=answer))

    # 7. Check limit and summarize if necessary
    if len(recent_msgs) >= MAX_RECENT_MESSAGES:
        # Update summary with these messages
        new_summary = summarize_messages(summary, recent_msgs)
        memory_store["conversation_summary"] = new_summary
        # Clear recent messages after merging into summary
        memory_store["recent_messages"] = []
    else:
        memory_store["recent_messages"] = recent_msgs

    # print("this is the summery")
    # print(summary)
    # print("this is the chat history",history_text)
    # print("this is the context")
    # print(context)
    return answer


# def start_chat():
#     print("Ask me questions! Type 'quit' to exit.")

#     while True:
#         question = input("\nYour question: ")

#         if question.lower() == "quit":
#             print("Goodbye!")
#             break

#         print("Gemini answer is-",ask_question(question))

# # -------------------------------
# # Entry point
# # -------------------------------
# if __name__ == "__main__":
#     start_chat()

# # ---------------------------------
# # FastAPI Setup
# # ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    question: str

@app.post("/chat")
async def chat(body: RequestBody):
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, ask_question, body.question)
    return {"reply": reply}






