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

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY or PINECONE_API_KEY")

INDEX_NAME = "gemini-rag-index2"
NAMESPACE = "default"

# ---------------------------------
# Gemini Embeddings (QUERY SIDE)
# ---------------------------------
# IMPORTANT: RETRIEVAL_QUERY is REQUIRED here
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
)

# ---------------------------------
# Pinecone Setup
# ---------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------
# Gemini LLM
# ---------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# ⚠️ Global memory (OK for demo, NOT for production)
chat_history = []

# ---------------------------------
# RAG Logic
# ---------------------------------
def ask_question(user_question: str) -> str:
    greetings = {"hi", "hello", "hey", "good morning"}
    if user_question.lower().strip() in greetings:
        return "Hello, I am Todung. How can I help you?"

    # Rewrite question if history exists
    if chat_history:
        history_text = "\n".join(msg.content for msg in chat_history)
        rewrite_prompt = (
            "Rewrite the following question to be standalone.\n\n"
            f"History:\n{history_text}\n\n"
            f"Question: {user_question}"
        )
        search_question = llm.invoke(rewrite_prompt).content.strip()
    else:
        search_question = user_question

    # Retrieve context
    docs = retriever.invoke(search_question)

    if not docs:
        return "I don't have enough information."

    context = "\n".join(doc.page_content for doc in docs)

    final_prompt = f"""
You are Todung, a helpful assistant.

Rules:
- Answer in ONE short sentence.
- Use ONLY the provided context.
- If the answer is not present, say: I don't have enough information.

Context:
{context}

User Question:
{user_question}
"""

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer

# ---------------------------------
# FastAPI Setup
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
