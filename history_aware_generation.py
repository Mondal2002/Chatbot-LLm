import os
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
# Embeddings (Sentence Transformers)
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
# LLaMA via Ollama
# -------------------------------
llm = OllamaLLM(
    model="tinyllama"   # change to "llama3" if you have it
)

# -------------------------------
# Conversation memory
# -------------------------------
chat_history = []

# -------------------------------
# Ask question function
# -------------------------------
def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question using 
    if chat_history:
        history_text = ""
        for msg in chat_history:
            history_text += f"- {msg.content}\n"

        rewrite_prompt = f"""
Given the chat history below, rewrite the new question so that it is standalone and searchable.
Return ONLY the rewritten question.

Chat history:
{history_text}

New question:
{user_question}
"""

        search_question = llm.invoke(rewrite_prompt).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview_lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(preview_lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build final context (SAFE)
    context = ""
    for doc in docs:
        context += f"- {doc.page_content}\n"

    final_prompt = f"""
You are a helpful assistant that answers questions ONLY using the provided documents.
If the answer is not present, say:
"I don't have enough information to answer that question based on the provided documents."

Documents:
{context}

Question:
{user_question}

Answer:
"""

    answer = llm.invoke(final_prompt).strip()

    # Step 4: Save conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# -------------------------------
# Chat loop
# -------------------------------
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    start_chat()
