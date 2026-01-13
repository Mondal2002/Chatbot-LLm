from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import asyncio
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# -------------------------------
# Environment
# -------------------------------
PINECONE_API_KEY = "pcsk_4vS2VH_3Z5Ck19AqgWSNebcJSyCpaRsVBddS4BrW1shwwzj65VLPyivimbLsD261utLoft"
INDEX_NAME = "chatbot-index"
NAMESPACE = "default"

# -------------------------------
# Embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

db = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE
)

# -------------------------------
# LLM (Mistral)
# -------------------------------
llm = OllamaLLM(
    model="phi3",
    temperature=0.2
)

# -------------------------------
# Conversation memory (TEMP – single user)
# -------------------------------
chat_history = []

# -------------------------------
# Blocking RAG function
# -------------------------------
def ask_question(user_question: str) -> str:
    # Rewrite question if history exists
    if chat_history:
        history_text = "\n".join(
            f"- {msg.content}" for msg in chat_history
        )

        rewrite_prompt = f"""
Rewrite the new question so it is standalone.
Return ONLY the rewritten question.

Chat history:
{history_text}

New question:
{user_question}
"""
        search_question = llm.invoke(rewrite_prompt).strip()
    else:
        search_question = user_question

    # Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    context = "\n".join(f"- {doc.page_content}" for doc in docs)

    # Final prompt (CLEAN)
    final_prompt = f"""
You are Todung, a helpful assistant.

Rules:
- If the user greets (hi, hello, hey, good morning),
  reply exactly:
  Hello, I am Todung. How can I help you?

- Answer in ONE short sentence.
- If you do not know the answer, say:
  I don't have enough information to answer that question.

Context:
{context}

User question:
{user_question}

Answer:
"""

    answer = llm.invoke(final_prompt).strip()

    # Save history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer

# -------------------------------
# FastAPI
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

    response_text = await loop.run_in_executor(
        None,
        ask_question,
        body.question
    )

    return {"reply": response_text}
