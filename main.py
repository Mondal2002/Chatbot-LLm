import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import (
    # ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.messages import HumanMessage, AIMessage
import boto3
import json
from botocore.exceptions import ClientError

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "gemini-rag-index4"
NAMESPACE = "default"

# ---------------------------------
# Gemini Embeddings & LLM
# ---------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    task_type="RETRIEVAL_QUERY"
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", # Updated to current stable version
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2,
# )

def invoke_mistral(
    prompt: str,
    region: str = "us-east-1",
    model_id: str = "mistral.mistral-large-2402-v1:0",
    max_tokens: int = 512,
    temperature: float = 0.5,
) -> str:
    """
    Invoke Mistral Large via AWS Bedrock and return the generated text.

    :param prompt: User prompt text
    :param region: AWS region where Bedrock is enabled
    :param model_id: Bedrock model ID
    :param max_tokens: Maximum tokens to generate
    :param temperature: Sampling temperature
    :return: Generated text response
    """

    client = boto3.client("bedrock-runtime", region_name=region)

    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    request_body = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
    except (ClientError, Exception) as e:
        raise RuntimeError(f"Failed to invoke model '{model_id}': {e}")

    model_response = json.loads(response["body"].read())
    return model_response["outputs"][0]["text"]
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
    response = invoke_mistral( summary_prompt)
    return response.strip()


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
    search_question = invoke_mistral(rewrite_prompt).strip()

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

    response = invoke_mistral(final_prompt)
    answer = response.strip()

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

# ---------------------------------
# FastAPI Setup
# ---------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://chatbot-launch-in4lj53x7-subham-mondals-projects-4d6994ac.vercel.app","https://chatbot-launch.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    question: str

@app.post("/chat")
async def chat(body: RequestBody):
    print("request recieved")
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, ask_question, body.question)
    print(reply)
    return {"reply": reply}
