import os
import json
import chromadb
import requests


##EDIT 1

# Constants
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "foods"
JSON_FILE = "foods.json"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"

# Load data
with open(JSON_FILE, "r", encoding="utf-8") as f:
    food_data = json.load(f)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Ollama embedding function
def get_embedding(text):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    return response.json()["embedding"]

# Add only new items
existing_ids = set(collection.get()['ids'])
new_items = [item for item in food_data if item['id'] not in existing_ids]

if new_items:
    print(f"🆕 Adding {len(new_items)} new documents to Chroma...")
    for item in new_items:
        # Enhance text with region/type
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."

        emb = get_embedding(enriched_text)

        collection.add(
            documents=[item["text"]],  # Use original text as retrievable context
            embeddings=[emb],
            ids=[item["id"]]
        )
else:
    print("✅ All documents already in ChromaDB.")

# RAG query
def rag_query(question):
    # Step 1: Embed the user question
    q_emb = get_embedding(question)

    # Step 2: Query the vector DB
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # Step 3: Extract documents
    top_docs = results['documents'][0]
    top_ids = results['ids'][0]

    # Step 4: Show friendly explanation of retrieved documents
    print("\n🧠 Retrieving relevant information to reason through your question...\n")

    for i, doc in enumerate(top_docs):
        print(f"🔹 Source {i + 1} (ID: {top_ids[i]}):")
        print(f"    \"{doc}\"\n")

    print("📚 These seem to be the most relevant pieces of information to answer your question.\n")

    # Step 5: Build prompt from context
    context = "\n".join(top_docs)

    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    # Step 6: Generate answer with Ollama
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })

    # Step 7: Return final result
    return response.json()["response"].strip()


# Interactive loop
print("\n🧠 RAG is ready. Ask a question (type 'exit' to quit):\n")
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    answer = rag_query(question)
    print("🤖:", answer)
