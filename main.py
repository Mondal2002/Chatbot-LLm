
import os
import asyncio
import json
import boto3
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from botocore.exceptions import ClientError

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aws import BedrockEmbeddings


# ---------------------------------
# Environment
# ---------------------------------
load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "titan-rag-index"
NAMESPACE = "default"
EMBEDDING_DIM = 1024

# ---------------------------------
# Threading
# ---------------------------------
answer_executor = ThreadPoolExecutor(max_workers=1)
summary_executor = ThreadPoolExecutor(max_workers=1)
bedrock_lock = threading.Lock()

# ---------------------------------
# Embeddings
# ---------------------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)

# ---------------------------------
# Bedrock – Mistral
# ---------------------------------
def invoke_mistral(prompt: str) -> str:
    with bedrock_lock:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        body = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": 512,
            "temperature": 0.5,
        }

        response = client.invoke_model(
            modelId="mistral.mistral-large-2402-v1:0",
            body=json.dumps(body),
        )

        payload = json.loads(response["body"].read())
        return payload["outputs"][0]["text"].strip()

# ---------------------------------
# Pinecone
# ---------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------
# Memory (PER USER, IN-MEMORY)
# ---------------------------------
MAX_RECENT_MESSAGES = 8
MAX_SUMMARIES = 3
MAX_SUMMARY_CHARS = 1200

memory_store = {}   # user_id -> memory
memory_locks = {}   # user_id -> lock


def get_user_memory(user_id: str):
    if user_id not in memory_store:
        memory_store[user_id] = {
            "summaries": [],
            "recent_messages": [],
        }
        memory_locks[user_id] = threading.Lock()

    return memory_store[user_id], memory_locks[user_id]

# ---------------------------------
# Background Summarization
# ---------------------------------
def run_summarization(user_id: str):
    memory, lock = get_user_memory(user_id)

    with lock:
        recent = memory["recent_messages"][:]
        summaries = memory["summaries"][:]

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent
    )

    summaries_text = "\n".join(summaries)

    prompt = f"""
Previous summaries:
{summaries_text}

New conversation:
{history_text}

Create a concise factual summary capturing only important context.
"""

    summary = invoke_mistral(prompt)[:MAX_SUMMARY_CHARS]

    with lock:
        summaries.append(summary)
        summaries[:] = summaries[-MAX_SUMMARIES:]
        memory["recent_messages"] = []

# ---------------------------------
# Main RAG Logic
# ---------------------------------
def ask_question(user_id: str, user_question: str) -> str:
    memory, lock = get_user_memory(user_id)

    with lock:
        summaries = memory["summaries"][:]
        recent = memory["recent_messages"][:]

    summaries_text = "\n".join(summaries)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent
    )

    docs = retriever.invoke(user_question)
    context = "\n".join(d.page_content for d in docs) if docs else ""

    final_prompt = f"""
You are Todung, a helpful assistant.

Rules:
- Greet only if user greets
- Reply in the SAME language as the user's question
- Answer in ONE short sentence
- Use context if relevant
- Otherwise say: I don't have enough information.

Conversation summaries:
{summaries_text}

Recent history:
{history_text}

Context:
{context}

User question:
{user_question}
"""

    answer = invoke_mistral(final_prompt)

    with lock:
        memory["recent_messages"].append(HumanMessage(content=user_question))
        memory["recent_messages"].append(AIMessage(content=answer))

        if len(memory["recent_messages"]) >= MAX_RECENT_MESSAGES:
            summary_executor.submit(run_summarization, user_id)

    return answer

# ---------------------------------
# FastAPI
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://chatbot-launch.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    user_id: str
    question: str

@app.post("/chat")
async def chat(body: RequestBody):
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(
        answer_executor,
        ask_question,
        body.user_id,
        body.question
    )
    print(body.user_id)
    return {"reply": answer}

