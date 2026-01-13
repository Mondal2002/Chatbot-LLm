import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone

# -------------------------------
# Environment & API Keys
# -------------------------------
# Replace with your actual Gemini API Key or set as environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBwEnNRc6aNKXhDyH7AqH6rMySbSj60wSc"
PINECONE_API_KEY = "pcsk_4vS2VH_3Z5Ck19AqgWSNebcJSyCpaRsVBddS4BrW1shwwzj65VLPyivimbLsD261utLoft" 
INDEX_NAME = "chatbot-index"
NAMESPACE = "default"

# -------------------------------
# Embeddings & Pinecone Setup
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

db = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)

# -------------------------------
# LLM (Gemini 2.5 Flash)
# -------------------------------
# Using Flash for speed and cost-efficiency
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chat_history = []

# -------------------------------
# Blocking RAG function
# -------------------------------
def ask_question(user_question: str) -> str:
    # 1. Handle Greetings Manually or via Prompt
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    if user_question.lower().strip() in greetings:
        return "Hello, I am Todung. How can I help you?"

    # 2. Rewrite question if history exists
    if chat_history:
        history_text = "\n".join(f"- {msg.content}" for msg in chat_history)
        rewrite_prompt = f"""Rewrite the new question to be standalone based on the chat history.
Return ONLY the rewritten question.

History:
{history_text}

New question: {user_question}"""
        
        # Gemini invoke
        search_result = llm.invoke(rewrite_prompt)
        search_question = search_result.content.strip()
    else:
        search_question = user_question

    # 3. Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    context = "\n".join(f"- {doc.page_content}" for doc in docs)

    # 4. Final Prompt
    final_prompt = f"""You are Todung, a helpful assistant.

Rules:
- Answer in ONE short sentence.
- Use the provided Context to answer.
- If you do not know the answer, say: I don't have enough information to answer that question.

Context:
{context}

User question:
{user_question}

Answer:"""

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # Save history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer

# -------------------------------
# FastAPI Setup
# -------------------------------
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
    response_text = await loop.run_in_executor(None, ask_question, body.question)
    return {"reply": response_text}