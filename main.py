
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

# # ---------------------------------
# # Environment
# # ---------------------------------
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# INDEX_NAME = "gemini-rag-index4"
# NAMESPACE = "default"

# # ---------------------------------
# # Threading & Locks
# # ---------------------------------
# answer_executor = ThreadPoolExecutor(max_workers=1)
# summary_executor = ThreadPoolExecutor(max_workers=1)

# memory_lock = threading.Lock()
# bedrock_lock = threading.Lock()

# # ---------------------------------
# # Embeddings
# # ---------------------------------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=GOOGLE_API_KEY,
#     task_type="RETRIEVAL_QUERY",
# )

# # ---------------------------------
# # Bedrock – Mistral
# # ---------------------------------
# def invoke_mistral(prompt: str) -> str:
#     with bedrock_lock:
#         client = boto3.client("bedrock-runtime", region_name="us-east-1")

#         body = {
#             "prompt": f"<s>[INST] {prompt} [/INST]",
#             "max_tokens": 512,
#             "temperature": 0.5,
#         }

#         try:
#             response = client.invoke_model(
#                 modelId="mistral.mistral-large-2402-v1:0",
#                 body=json.dumps(body),
#             )
#         except (ClientError, Exception) as e:
#             raise RuntimeError(f"Bedrock invocation failed: {e}")

#         payload = json.loads(response["body"].read())
#         return payload["outputs"][0]["text"].strip()

# # ---------------------------------
# # Pinecone
# # ---------------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# vectorstore = PineconeVectorStore(
#     index=index,
#     embedding=embeddings,
#     namespace=NAMESPACE,
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ---------------------------------
# # Memory
# # ---------------------------------
# MAX_RECENT_MESSAGES = 8
# MAX_SUMMARIES = 3
# MAX_SUMMARY_CHARS = 1200

# memory_store = {}          # user_id -> memory
# memory_locks = {}          # user_id -> lock

# # ---------------------------------
# # Background Summarization
# # ---------------------------------
# def run_summarization():
#     with lock:
#         summaries.append(summary)
#         summaries[:] = summaries[-MAX_SUMMARIES:]
#         memory["recent_messages"] = []


#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent
#     )

#     summaries_text = "\n".join(summaries)

#     prompt = f"""
# Previous summaries:
# {summaries_text}

# New conversation:
# {history_text}

# Create a concise factual summary capturing only important context.
# """

#     summary = invoke_mistral(prompt)[:MAX_SUMMARY_CHARS]

#     with memory_lock:
#         summaries.append(summary)
#         if len(summaries) > MAX_SUMMARIES:
#             summaries.pop(0)

#         memory_store["summaries"] = summaries
#         memory_store["recent_messages"] = []


# def get_user_memory(user_id: str):
#     if user_id not in memory_store:
#         memory_store[user_id] = {
#             "summaries": [],
#             "recent_messages": [],
#         }
#         memory_locks[user_id] = threading.Lock()

#     return memory_store[user_id], memory_locks[user_id]


# # ---------------------------------
# # Main RAG Logic
# # ---------------------------------
# def ask_question(user_question: str) -> str:
#     memory, lock = get_user_memory(user_id)
#     with lock:
#         summaries = memory_store["summaries"][:]
#         recent = memory_store["recent_messages"][:]

#     summaries_text = "\n".join(summaries)
#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent
#     )

# #     rewrite_prompt = f"""
# # Conversation summaries:
# # {summaries_text}

# # Recent history:
# # {history_text}

# # Question:
# # {user_question}

# # Rewrite as a standalone search query.
# # """
#     search_query = user_question

#     docs = retriever.invoke(search_query)
#     context = "\n".join(d.page_content for d in docs) if docs else ""

#     final_prompt = f"""
# You are Todung, a helpful assistant.

# Rules:
# - Greet only if user greets
# - Answer in ONE short sentence
# - Use context if relevant
# - Otherwise say: I don't have enough information.

# Conversation summaries:
# {summaries_text}

# Recent history:
# {history_text}

# Context:
# {context}

# User question:
# {user_question}
# """
#     answer = invoke_mistral(final_prompt)

#     with lock:
#         memory["recent_messages"].append(HumanMessage(content=user_question))
#         memory["recent_messages"].append(AIMessage(content=answer))


#         if len(memory_store["recent_messages"]) >= MAX_RECENT_MESSAGES:
#             summary_executor.submit(run_summarization, user_id)


#     return answer

# # ---------------------------------
# # FastAPI
# # ---------------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "https://chatbot-launch.vercel.app",
#     ],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class RequestBody(BaseModel):
#     user_id: str
#     question: str

# @app.post("/chat")
# async def chat(body: RequestBody):
#     loop = asyncio.get_running_loop()
#     answer = await loop.run_in_executor(
#         answer_executor, ask_question, body.user_id, body.question
#     )

#     return {"reply": answer}




# import os
# import asyncio
# import json
# import boto3
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from botocore.exceptions import ClientError

# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.messages import HumanMessage, AIMessage

# # ---------------------------------
# # Load environment variables
# # ---------------------------------
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# INDEX_NAME = "gemini-rag-index4"
# NAMESPACE = "default"

# # ---------------------------------
# # Embeddings
# # ---------------------------------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=GOOGLE_API_KEY,
#     task_type="RETRIEVAL_QUERY",
# )

# # ---------------------------------
# # AWS Bedrock – Mistral
# # ---------------------------------
# def invoke_mistral(
#     prompt: str,
#     region: str = "us-east-1",
#     model_id: str = "mistral.mistral-large-2402-v1:0",
#     max_tokens: int = 512,
#     temperature: float = 0.5,
# ) -> str:

#     client = boto3.client("bedrock-runtime", region_name=region)

#     request_body = {
#         "prompt": f"<s>[INST] {prompt} [/INST]",
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#     }

#     try:
#         response = client.invoke_model(
#             modelId=model_id,
#             body=json.dumps(request_body),
#         )
#     except (ClientError, Exception) as e:
#         raise RuntimeError(f"Bedrock invocation failed: {e}")

#     body = json.loads(response["body"].read())
#     return body["outputs"][0]["text"].strip()

# # ---------------------------------
# # Pinecone Setup
# # ---------------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# vectorstore = PineconeVectorStore(
#     index=index,
#     embedding=embeddings,
#     namespace=NAMESPACE,
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ---------------------------------
# # Memory Configuration
# # ---------------------------------
# MAX_RECENT_MESSAGES = 8     # rolling buffer
# MAX_SUMMARIES = 3
# MAX_SUMMARY_CHARS = 1200

# memory_store = {
#     "summaries": [],        # list[str]
#     "recent_messages": [], # list[BaseMessage]
# }

# # ---------------------------------
# # Summarization
# # ---------------------------------
# def summarize_messages(previous_summaries: list[str], recent_msgs: list) -> str:
#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent_msgs
#     )

#     summaries_text = "\n".join(previous_summaries)

#     prompt = f"""
# Previous summaries:
# {summaries_text}

# New conversation:
# {history_text}

# Create a concise factual summary capturing only key information and decisions.
# """

#     summary = invoke_mistral(prompt)
#     return summary[:MAX_SUMMARY_CHARS]

# # ---------------------------------
# # Main RAG Logic
# # ---------------------------------
# def ask_question(user_question: str) -> str:
#     summaries = memory_store["summaries"]
#     recent_msgs = memory_store["recent_messages"]

#     summaries_text = "\n".join(summaries)
#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent_msgs
#     )

#     # 1. Rewrite question
#     rewrite_prompt = f"""
# Conversation summaries:
# {summaries_text}

# Recent history:
# {history_text}

# Question:
# {user_question}

# Rewrite as a standalone search query.
# """
#     search_query = invoke_mistral(rewrite_prompt)

#     # 2. Retrieve context
#     docs = retriever.invoke(search_query)
#     context = "\n".join(d.page_content for d in docs) if docs else ""

#     # 3. Answer
#     final_prompt = f"""
# You are Todung, a helpful assistant.

# Rules:
# - Greet only if the user greets
# - Answer in ONE short sentence
# - Use provided context
# - If unsure, say: I don't have enough information.

# Conversation summaries:
# {summaries_text}

# Recent history:
# {history_text}

# Context:
# {context}

# User question:
# {user_question}
# """
#     answer = invoke_mistral(final_prompt)

#     # 4. Update memory
#     recent_msgs.append(HumanMessage(content=user_question))
#     recent_msgs.append(AIMessage(content=answer))

#     if len(recent_msgs) >= MAX_RECENT_MESSAGES:
#         new_summary = summarize_messages(summaries, recent_msgs)
#         summaries.append(new_summary)

#         if len(summaries) > MAX_SUMMARIES:
#             summaries.pop(0)

#         memory_store["recent_messages"] = []
#         memory_store["summaries"] = summaries

#     return answer

# # ---------------------------------
# # FastAPI Setup
# # ---------------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "https://chatbot-launch.vercel.app",
#     ],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class RequestBody(BaseModel):
#     question: str

# @app.post("/chat")
# async def chat(body: RequestBody):
#     loop = asyncio.get_running_loop()
#     reply = await loop.run_in_executor(None, ask_question, body.question)
#     return {"reply": reply}





# import os
# import asyncio
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv

# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import (
#     # ChatGoogleGenerativeAI,
#     GoogleGenerativeAIEmbeddings
# )
# from langchain_core.messages import HumanMessage, AIMessage
# import boto3
# import json
# from botocore.exceptions import ClientError

# # ---------------------------------
# # Load environment variables
# # ---------------------------------
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# INDEX_NAME = "gemini-rag-index4"
# NAMESPACE = "default"

# # ---------------------------------
# # Gemini Embeddings & LLM
# # ---------------------------------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=GOOGLE_API_KEY,
#     task_type="RETRIEVAL_QUERY"
# )

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.5-flash", # Updated to current stable version
# #     google_api_key=GOOGLE_API_KEY,
# #     temperature=0.2,
# # )

# def invoke_mistral(
#     prompt: str,
#     region: str = "us-east-1",
#     model_id: str = "mistral.mistral-large-2402-v1:0",
#     max_tokens: int = 512,
#     temperature: float = 0.5,
# ) -> str:
#     """
#     Invoke Mistral Large via AWS Bedrock and return the generated text.

#     :param prompt: User prompt text
#     :param region: AWS region where Bedrock is enabled
#     :param model_id: Bedrock model ID
#     :param max_tokens: Maximum tokens to generate
#     :param temperature: Sampling temperature
#     :return: Generated text response
#     """

#     client = boto3.client("bedrock-runtime", region_name=region)

#     formatted_prompt = f"<s>[INST] {prompt} [/INST]"

#     request_body = {
#         "prompt": formatted_prompt,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#     }

#     try:
#         response = client.invoke_model(
#             modelId=model_id,
#             body=json.dumps(request_body)
#         )
#     except (ClientError, Exception) as e:
#         raise RuntimeError(f"Failed to invoke model '{model_id}': {e}")

#     model_response = json.loads(response["body"].read())
#     return model_response["outputs"][0]["text"]
# # ---------------------------------
# # Pinecone Setup
# # ---------------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)
# vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ---------------------------------
# # Memory Management State
# # ---------------------------------
# # In production, use a dictionary keyed by session_id
# memory_store = {
#     "conversation_summary": "",
#     "recent_messages": [], # List of BaseMessages
# }
# # MAX_RECENT_MESSAGES = 8  # N turns (3 Human + 3 AI)

# # def summarize_messages(summary: str, recent_msgs: list) -> str:
# #     """Summarizes recent messages and merges them with the existing summary."""
# #     history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_msgs])
    
# #     summary_prompt = (
# #         f"Current Summary: {summary}\n\n"
# #         f"New Messages to incorporate:\n{history_text}\n\n"
# #         "Generate a concise updated summary of the conversation so far."
# #     )
# #     response = invoke_mistral( summary_prompt)
# #     return response.strip()

# MAX_RECENT_MESSAGES = 8          # 3 Human + 3 AI
# MAX_SUMMARIES = 3
# MAX_SUMMARY_CHARS = 1200

# def summarize_messages(previous_summaries: list[str], recent_msgs: list) -> str:
#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent_msgs
#     )

#     summaries_text = "\n".join(previous_summaries)

#     prompt = (
#         f"Previous conversation summaries:\n{summaries_text}\n\n"
#         f"New messages:\n{history_text}\n\n"
#         "Create a concise summary capturing only important facts, decisions, and context."
#     )

#     summary = invoke_mistral(prompt).strip()
#     return summary[:MAX_SUMMARY_CHARS]


# # ---------------------------------
# # RAG Logic
# # ---------------------------------
# # def ask_question(user_question: str) -> str:
# #     global memory_store

# #     # 1. Load session memory
# #     summary = memory_store["conversation_summary"]
# #     recent_msgs = memory_store["recent_messages"]

# #     # 2. Build History Text for Query Rewriting/Context
# #     history_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_msgs])
    
# #     # 3. Rewrite question to be standalone (Context-Aware)
# #     rewrite_prompt = (
# #         f"Summary of conversation: {summary}\n"
# #         f"Recent History: {history_text}\n"
# #         f"Question: {user_question}\n"
# #         "Given the history and summary, rewrite the question to be a standalone search query."
# #     )
# #     search_question = invoke_mistral(rewrite_prompt).strip()

# #     # 4. Retrieve context from Pinecone
# #     docs = retriever.invoke(search_question)
# #     context = "\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."

# #     # 5. Build final prompt with Summary + Recent + Context
# #     final_prompt = f"""
# # You are Todung, a helpful assistant.
# # Rules:
# # - For greetings (hi, hello, etc.), reply: "Hello, I am Todung. How can I help you?"
# # - Answer in ONE short sentence.
# # - Use the provided Context and Conversation Summary to inform your answer.
# # - If the answer is not in the context, say: I don't have enough information.

# # Conversation Summary: {summary}
# # Recent History: {history_text}
# # Context: {context}

# # User Question: {user_question}
# # """

# #     response = invoke_mistral(final_prompt)
# #     answer = response.strip()

# #     # 6. Update recent_messages
# #     recent_msgs.append(HumanMessage(content=user_question))
# #     recent_msgs.append(AIMessage(content=answer))

# #     # 7. Check limit and summarize if necessary
# #     if len(recent_msgs) >= MAX_RECENT_MESSAGES:
# #         # Update summary with these messages
# #         new_summary = summarize_messages(summary, recent_msgs)
# #         memory_store["conversation_summary"] = new_summary
# #         # Clear recent messages after merging into summary
# #         memory_store["recent_messages"] = []
# #     else:
# #         memory_store["recent_messages"] = recent_msgs

# #     # print("this is the summery")
# #     # print(summary)
# #     # print("this is the chat history",history_text)
# #     # print("this is the context")
# #     # print(context)
# #     return answer
# def ask_question(user_question: str) -> str:
#     global memory_store

#     summaries = memory_store["summaries"]
#     recent_msgs = memory_store["recent_messages"]

#     summaries_text = "\n".join(summaries)

#     history_text = "\n".join(
#         f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
#         for m in recent_msgs
#     )

#     # 1. Rewrite question
#     rewrite_prompt = (
#         f"Conversation summaries:\n{summaries_text}\n"
#         f"Recent history:\n{history_text}\n"
#         f"Question: {user_question}\n"
#         "Rewrite this as a standalone search query."
#     )
#     search_question = invoke_mistral(rewrite_prompt).strip()

#     # 2. Retrieve RAG context
#     docs = retriever.invoke(search_question)
#     context = "\n".join(d.page_content for d in docs) if docs else "No relevant context found."

#     # 3. Final answer prompt
#     final_prompt = f"""
# You are Todung, a helpful assistant.
# Rules:
# - Greet only if user greets
# - Answer in ONE short sentence
# - Use summaries and context
# - If insufficient info, say: I don't have enough information.

# Conversation Summaries:
# {summaries_text}

# Recent History:
# {history_text}

# Context:
# {context}

# User Question:
# {user_question}
# """

#     answer = invoke_mistral(final_prompt).strip()

#     # 4. Update recent memory
#     recent_msgs.append(HumanMessage(content=user_question))
#     recent_msgs.append(AIMessage(content=answer))

#     # 5. Summarize if buffer full
#     if len(recent_msgs) >= MAX_RECENT_MESSAGES:
#         new_summary = summarize_messages(summaries, recent_msgs)

#         summaries.append(new_summary)
#         if len(summaries) > MAX_SUMMARIES:
#             summaries.pop(0)  # forget oldest summary

#         memory_store["recent_messages"] = []
#         memory_store["summaries"] = summaries
#     else:
#         memory_store["recent_messages"] = recent_msgs

#     return answer

# # ---------------------------------
# # FastAPI Setup
# # ---------------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000","https://chatbot-launch-in4lj53x7-subham-mondals-projects-4d6994ac.vercel.app","https://chatbot-launch.vercel.app"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class RequestBody(BaseModel):
#     question: str

# @app.post("/chat")
# async def chat(body: RequestBody):
#     print("request recieved")
#     loop = asyncio.get_running_loop()
#     reply = await loop.run_in_executor(None, ask_question, body.question)
#     print(reply)
#     return {"reply": reply}
