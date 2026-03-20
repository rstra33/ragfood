# -*- coding: utf-8 -*-
import os
import json
import time
import sys
import requests
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq
from groq._exceptions import RateLimitError, APIStatusError, AuthenticationError

# Fix for Windows console Unicode
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
JSON_FILE  = "foods.json"
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── Upstash Vector clients ───────────────────────────────────────────────────
def _make_index(readonly=False):
    url   = os.getenv("UPSTASH_VECTOR_REST_URL")
    token = (
        os.getenv("UPSTASH_VECTOR_REST_READONLY_TOKEN")
        if readonly
        else os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )
    if not url or not token:
        raise ValueError(
            f"Missing Upstash credentials in .env "
            f"({'READONLY_TOKEN' if readonly else 'TOKEN'} or URL)"
        )
    return Index(url=url, token=token)

admin_index = _make_index(readonly=False)
query_index = _make_index(readonly=True)

# ─── Groq client ──────────────────────────────────────────────────────────────
def _make_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=key)

groq_client = _make_groq()

# ─── Startup validation ───────────────────────────────────────────────────────
def validate_services():
    """Quick connectivity check for both cloud services."""
    ok = True

    # Upstash: attempt a dummy query (empty index is OK — just checking auth)
    try:
        query_index.query(data="test", top_k=1, include_metadata=True)
        print("✅ Upstash Vector: connected")
    except Exception as e:
        print(f"❌ Upstash Vector: {e}")
        ok = False

    # Groq: minimal 1-token completion
    try:
        groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=1
        )
        print("✅ Groq API: connected")
    except Exception as e:
        print(f"❌ Groq API: {e}")
        ok = False

    return ok

# ─── Data ingestion ───────────────────────────────────────────────────────────
def ingest_food_data(food_data):
    """
    Upsert food items into Upstash Vector with retry.
    Upstash auto-embeds text using mxbai-embed-large-v1 (1024 dims).
    Upsert is idempotent — safe to re-run without duplicating entries.
    """
    vectors = []
    for item in food_data:
        # Enrich text for better semantic search (same logic as original)
        enriched = item["text"]
        if "region" in item:
            enriched += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched += f" It is a type of {item['type']}."

        vectors.append((
            item["id"],
            enriched,                           # text Upstash will auto-embed
            {
                "original_text": item["text"],  # stored clean for retrieval
                "region": item.get("region", "Unknown"),
                "type":   item.get("type", "Unknown"),
            }
        ))

    print(f"🆕 Upserting {len(vectors)} documents to Upstash...")
    for attempt in range(3):
        try:
            admin_index.upsert(vectors=vectors)
            print(f"✅ {len(vectors)} documents indexed.")
            return True
        except Exception as e:
            err = str(e)
            if "401" in err or "400" in err:
                print(f"❌ Upsert failed (non-retryable): {err}")
                return False
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️  Upsert error, retrying in {wait}s... ({err})")
                time.sleep(wait)
            else:
                print(f"❌ Upsert failed after 3 attempts: {err}")
                return False

# ─── Vector query with retry ──────────────────────────────────────────────────
def _vector_query(question, top_k=3):
    """Query Upstash with exponential backoff. Returns list of doc strings."""
    for attempt in range(3):
        try:
            results = query_index.query(
                data=question, top_k=top_k, include_metadata=True
            )
            return [r.metadata["original_text"] for r in results] if results else []        
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                print("❌ Upstash query: auth failed. Check READONLY_TOKEN in .env")
                return []
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️  Upstash query error, retrying in {wait}s... ({err})")
                time.sleep(wait)
            else:
                print(f"❌ Upstash query failed after 3 attempts: {err}")
                return []

# ─── Groq completion with retry ───────────────────────────────────────────────
def _groq_completion(messages):
    """Call Groq with exponential backoff. Returns answer string."""
    for attempt in range(3):
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=512,
                top_p=0.95
            )
            usage = completion.usage
            print(f"📊 Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
            return completion.choices[0].message.content.strip()

        except AuthenticationError:
            print("❌ Groq: invalid API key. Check GROQ_API_KEY in .env")
            return "Configuration error: invalid Groq API key."

        except RateLimitError:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️  Groq rate limit (attempt {attempt+1}/3). Waiting {wait}s...")
                time.sleep(wait)
            else:
                return "Too many requests. Please try again in a moment."

        except APIStatusError as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️  Groq API error {e.status_code} (attempt {attempt+1}/3). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "Groq service is temporarily unavailable. Please try again soon."

        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                return f"Unexpected error: {str(e)}"

    return "Failed to generate a response after multiple retries."

# ─── Main RAG query ───────────────────────────────────────────────────────────
def rag_query(question):
    """
    Full RAG pipeline:
      1. Embed + search via Upstash (auto-embedding, with retry)
      2. Build prompt from retrieved context
      3. Generate answer via Groq (with retry)
    """
    # Step 1: retrieve context
    print("\n🧠 Retrieving relevant information...\n")
    top_docs = _vector_query(question)

    if top_docs is None:
        return "Could not reach the vector database. Please try again."

    if not top_docs:
        return "I couldn't find relevant information for that question."

    for i, doc in enumerate(top_docs):
        print(f"🔹 Source {i+1}: \"{doc[:100]}...\"")

    print("\n📚 Using the above context to generate an answer.\n")

    # Step 2: build messages (OpenAI chat format)
    context  = "\n".join(top_docs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about food "
                "using the provided context. Be concise and accurate. "
                "If the context does not contain enough information, say so."
            )
        },
        {
            "role": "user",
            "content": (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        }
    ]

    # Step 3: generate answer
    return _groq_completion(messages)

# ─── Optional: Ollama fallback for LLM ───────────────────────────────────────
def rag_query_ollama_fallback(question, top_docs):
    """
    Legacy Ollama LLM call. Used only when Groq is unavailable.
    Requires Ollama running locally at localhost:11434.
    """
    context = "\n".join(top_docs)
    prompt  = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Both Groq and Ollama failed. Last error: {str(e)}"

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Validate both services before starting
    if not validate_services():
        print("\n❌ One or more services failed to connect. Check your .env file.")
        exit(1)

    # Load and index data (idempotent — safe to run every startup)
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        food_data = json.load(f)

    if not ingest_food_data(food_data):
        print("\n❌ Failed to ingest food data. Exiting.")
        exit(1)

    # Interactive loop
    print("\n🧠 RAG is ready. Ask a question (type 'exit' to quit):\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        answer = rag_query(question)
        print(f"🤖: {answer}\n")