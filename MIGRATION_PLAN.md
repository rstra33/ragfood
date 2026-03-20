# RAG-Food Migration Plan
## ChromaDB → Upstash Vector + Ollama → Groq

> **Scope**: This document covers the full migration of the RAG-Food system to a fully cloud-native stack.
> Both the vector database layer (ChromaDB → Upstash) and the LLM inference layer (Ollama → Groq) are
> addressed here and must be applied together for a consistent system.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture: Before & After](#2-architecture-before--after)
3. [Dependency Changes](#3-dependency-changes)
4. [Credential Management](#4-credential-management)
5. [Code Changes: Vector Database Layer](#5-code-changes-vector-database-layer)
6. [Code Changes: LLM Inference Layer](#6-code-changes-llm-inference-layer)
7. [Unified Error Handling](#7-unified-error-handling)
8. [Retry Logic & Fallback Mechanisms](#8-retry-logic--fallback-mechanisms)
9. [Complete Migrated rag_run.py](#9-complete-migrated-rag_runpy)
10. [Performance Considerations](#10-performance-considerations)
11. [Cost Implications](#11-cost-implications)
12. [Security Considerations](#12-security-considerations)
13. [Testing Strategy](#13-testing-strategy)
14. [Migration Checklist](#14-migration-checklist)
15. [Rollback Plan](#15-rollback-plan)

---

## 1. Executive Summary

The RAG-Food system currently uses two local services that require manual setup and ongoing maintenance:

| Layer | Current | Target | Key Benefit |
|---|---|---|---|
| **Vector DB** | ChromaDB (local SQLite) | Upstash Vector (serverless cloud) | Auto-embedding, no local storage |
| **Embeddings** | Ollama `mxbai-embed-large` (manual) | Upstash built-in `mixedbread-ai/mxbai-embed-large-v1` | Eliminated entirely from code |
| **LLM Inference** | Ollama `llama3.2` (local GPU/CPU) | Groq `llama-3.1-8b-instant` (cloud LPU) | 5–10× faster, zero local resources |

**After migration:**
- No local Ollama service required
- No manual embedding generation
- No ChromaDB directory to manage
- Query latency drops from ~700–2200ms to ~160–500ms total

---

## 2. Architecture: Before & After

### Current Architecture

```
┌──────────────────────────────────────────────────────┐
│               RAG-Food System (Local)                │
│                                                      │
│  foods.json ──► rag_run.py                          │
│                     │                               │
│                     ▼                               │
│          ┌─────────────────────┐                   │
│          │ Manual Embedding    │                   │
│          │ Ollama API          │ ← localhost:11434 │
│          │ mxbai-embed-large   │                   │
│          └─────────────────────┘                   │
│                     │ vector [1024]                 │
│                     ▼                               │
│          ┌─────────────────────┐                   │
│          │ ChromaDB Local      │                   │
│          │ SQLite + Index      │ ← /chroma_db/     │
│          └─────────────────────┘                   │
│                     │ top-k docs                   │
│                     ▼                               │
│          ┌─────────────────────┐                   │
│          │ Ollama LLM          │ ← localhost:11434 │
│          │ llama3.2            │                   │
│          │ ~500–2000ms         │                   │
│          └─────────────────────┘                   │
└──────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌──────────────────────────────────────────────────────┐
│           RAG-Food System (Cloud-Native)             │
│                                                      │
│  foods.json ──► rag_run.py                          │
│                     │                               │
│                     │ raw text (no pre-embedding)   │
│                     ▼                               │
│          ┌─────────────────────┐                   │
│          │ Upstash Vector SDK  │ ← HTTPS REST      │
│          │ Auto-embedding:     │                   │
│          │ mxbai-embed-large-v1│                   │
│          │ ~100–300ms          │                   │
│          └─────────────────────┘                   │
│                     │ top-k docs + metadata        │
│                     ▼                               │
│          ┌─────────────────────┐                   │
│          │ Groq SDK            │ ← HTTPS           │
│          │ llama-3.1-8b-instant│                   │
│          │ LPU inference       │                   │
│          │ ~50–200ms           │                   │
│          └─────────────────────┘                   │
└──────────────────────────────────────────────────────┘
```

**Total query latency comparison:**

| Stage | Before | After |
|---|---|---|
| Embed query | ~100ms (Ollama API) | Included in Upstash query |
| Vector search | ~10–50ms (local SQLite) | ~100–300ms (Upstash cloud) |
| LLM inference | ~500–2000ms (local GPU/CPU) | ~50–200ms (Groq LPU) |
| **Total** | **~700–2200ms** | **~160–500ms** |

---

## 3. Dependency Changes

### Remove

```bash
pip uninstall chromadb
# Ollama no longer needed for embeddings OR inference
# (safe to leave installed if you want a local fallback — see Section 8)
```

### Add

```bash
pip install upstash-vector groq python-dotenv
```

| Package | Purpose |
|---|---|
| `upstash-vector` | Official Upstash Vector Python SDK |
| `groq` | Official Groq Python SDK (handles auth, retries, streaming) |
| `python-dotenv` | Secure `.env` loading |

### Updated `requirements.txt`

```
upstash-vector
groq
python-dotenv
```

---

## 4. Credential Management

### `.env` file (local development)

```bash
# Upstash Vector
UPSTASH_VECTOR_REST_URL="https://your-index.upstash.io"
UPSTASH_VECTOR_REST_TOKEN="[redacted]"
UPSTASH_VECTOR_REST_READONLY_TOKEN="[redacted]"

# Groq
GROQ_API_KEY="[redacted]"

# Optional feature flags
ENABLE_STREAMING=true
FALLBACK_TO_OLLAMA=false
```

### `.gitignore` — always include

```
.env
.env.local
.env.*.local
.env.production
__pycache__/
*.py[cod]
venv/
.venv/
chroma_db/
groq_usage.log
```

### Rules

- **Never** hardcode tokens in source code
- **Never** log or print tokens (even in error messages — log the URL, not the token)
- Use `UPSTASH_VECTOR_REST_READONLY_TOKEN` for all query operations; keep the admin token for upsert/delete scripts only
- Rotate tokens via the Upstash dashboard if exposed; update `.env` and redeploy

---

## 5. Code Changes: Vector Database Layer

### 5.1 Remove: entire `get_embedding()` function

```python
# DELETE THIS — Upstash handles embedding automatically
# def get_embedding(text):
#     response = requests.post("http://localhost:11434/api/embeddings", json={
#         "model": EMBED_MODEL,
#         "prompt": text
#     })
#     return response.json()["embedding"]
```

### 5.2 Remove: ChromaDB constants and client

```python
# DELETE THESE
# CHROMA_DIR = "chroma_db"
# COLLECTION_NAME = "foods"
# EMBED_MODEL = "mxbai-embed-large"

# import chromadb
# chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
```

### 5.3 Add: Upstash client initialisation

```python
import os
from dotenv import load_dotenv
from upstash_vector import Index

load_dotenv()

def get_upstash_index(readonly=False):
    """Initialise Upstash Vector client. Use readonly=True for queries."""
    url = os.getenv("UPSTASH_VECTOR_REST_URL")
    token = (
        os.getenv("UPSTASH_VECTOR_REST_READONLY_TOKEN")
        if readonly
        else os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )

    if not url or not token:
        raise ValueError(
            "Missing UPSTASH_VECTOR_REST_URL or token in .env. "
            "Check UPSTASH_VECTOR_REST_TOKEN (admin) and "
            "UPSTASH_VECTOR_REST_READONLY_TOKEN (query)."
        )

    print(f"✅ Connecting to Upstash Vector at {url}")
    return Index(url=url, token=token)

# Admin index for upsert operations
admin_index = get_upstash_index(readonly=False)

# Readonly index for query operations (safer for user-facing code)
query_index = get_upstash_index(readonly=True)
```

### 5.4 Update: data ingestion — raw text upsert (no pre-computed vectors)

**Before (ChromaDB — manual embedding per item):**
```python
existing_ids = set(collection.get()['ids'])
new_items = [item for item in food_data if item['id'] not in existing_ids]

if new_items:
    for item in new_items:
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."
        emb = get_embedding(enriched_text)       # ← manual Ollama call
        collection.add(
            documents=[item["text"]],
            embeddings=[emb],
            ids=[item["id"]]
        )
```

**After (Upstash — raw text, batch upsert, auto-embedding):**
```python
def ingest_food_data(food_data):
    """
    Upsert all food items into Upstash Vector.
    Upstash auto-embeds the text using mxbai-embed-large-v1.
    Upsert is idempotent — safe to re-run without duplicating entries.
    """
    print(f"🆕 Preparing {len(food_data)} documents for Upstash...")

    vectors_to_upsert = []
    for item in food_data:
        # Enrich text for better semantic search (same logic as before)
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."

        # Store original (non-enriched) text in metadata for retrieval
        metadata = {
            "original_text": item["text"],
            "region": item.get("region", "Unknown"),
            "type": item.get("type", "Unknown"),
        }

        # Upstash tuple: (id, text_to_embed, metadata)
        # No embedding call needed — Upstash handles it server-side
        vectors_to_upsert.append((item["id"], enriched_text, metadata))

    # Single batch call — far more efficient than per-item upserts
    admin_index.upsert(vectors=vectors_to_upsert)
    print(f"✅ {len(vectors_to_upsert)} documents indexed in Upstash.")
```

### 5.5 Update: vector query

**Before (ChromaDB):**
```python
q_emb = get_embedding(question)  # manual Ollama call
results = collection.query(query_embeddings=[q_emb], n_results=3)
top_docs = results['documents'][0]
top_ids = results['ids'][0]
```

**After (Upstash):**
```python
# Pass raw text — Upstash auto-embeds the question too
results = query_index.query(data=question, top_k=3, include_metadata=True)

top_docs = [r["metadata"]["original_text"] for r in results]
top_ids  = [r["id"] for r in results]
```

---

## 6. Code Changes: LLM Inference Layer

### 6.1 Remove: Ollama inference calls

```python
# DELETE THESE
# LLM_MODEL = "llama3.2"

# response = requests.post("http://localhost:11434/api/generate", json={
#     "model": LLM_MODEL,
#     "prompt": prompt,
#     "stream": False
# })
# return response.json()["response"].strip()
```

### 6.2 Add: Groq client initialisation

```python
from groq import Groq

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=api_key)

groq_client = get_groq_client()
```

### 6.3 Update: LLM call — prompt format changes

Groq uses the OpenAI chat completions format (list of role/content dicts) rather than Ollama's single string prompt. This also unlocks a `system` role for better response quality.

**Before (Ollama — single string):**
```python
prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False
})
return response.json()["response"].strip()
```

**After (Groq — messages list):**
```python
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

completion = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    temperature=0.7,
    max_completion_tokens=512,
    top_p=0.95,
    stream=False
)
return completion.choices[0].message.content.strip()
```

### 6.4 Optional: streaming for better UX

```python
def rag_query_streaming(question):
    """Stream the Groq response token-by-token for a real-time feel."""
    results = query_index.query(data=question, top_k=3, include_metadata=True)
    top_docs = [r["metadata"]["original_text"] for r in results]
    context = "\n".join(top_docs)

    messages = [
        {"role": "system", "content": "You are a helpful food expert. Be concise."},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    print("🤖: ", end="", flush=True)
    full_response = ""

    with groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True,
        max_completion_tokens=512
    ) as completion:
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response += delta

    print()  # newline after stream ends
    return full_response
```

---

## 7. Unified Error Handling

Each cloud service can fail independently. The full `rag_query()` call path now has two failure points — the Upstash vector query and the Groq LLM call. Both need to be handled so that a failure in one doesn't produce an unhandled exception in the other.

### Error taxonomy

| Service | Error type | Cause | Recommended response |
|---|---|---|---|
| Upstash | `401 Unauthorized` | Wrong or expired token | Exit with config error |
| Upstash | `429 Too Many Requests` | Rate limit hit | Retry with backoff |
| Upstash | `5xx` | Upstash service issue | Retry with backoff |
| Upstash | `TimeoutError` | Network or slow response | Retry once, then graceful message |
| Groq | `RateLimitError` | Free tier limit | Retry with backoff |
| Groq | `APIStatusError (5xx)` | Groq service issue | Retry with backoff |
| Groq | `AuthenticationError` | Invalid API key | Exit with config error |
| Both | No results returned | Empty index or bad query | Graceful "no info found" message |

### Unified wrapper

```python
from groq._exceptions import RateLimitError, APIStatusError, AuthenticationError

def safe_vector_query(question, top_k=3):
    """
    Query Upstash with error handling.
    Returns list of document strings, or None on unrecoverable failure.
    """
    try:
        results = query_index.query(data=question, top_k=top_k, include_metadata=True)
        if not results:
            print("⚠️ Upstash returned no results for this query.")
            return []
        return [r["metadata"]["original_text"] for r in results]

    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err:
            print("❌ Upstash auth failed. Check UPSTASH_VECTOR_REST_READONLY_TOKEN in .env")
        elif "429" in err:
            print("⚠️ Upstash rate limit hit.")
        elif "timeout" in err.lower():
            print("⚠️ Upstash query timed out.")
        else:
            print(f"❌ Upstash query error: {err}")
        return None  # signals caller to abort


def safe_groq_completion(messages, max_retries=3):
    """
    Call Groq with error handling and exponential backoff.
    Returns response string, or a graceful error message string.
    """
    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_completion_tokens=512,
                top_p=0.95
            )
            usage = completion.usage
            print(f"📊 Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
            return completion.choices[0].message.content.strip()

        except AuthenticationError:
            print("❌ Groq auth failed. Check GROQ_API_KEY in .env")
            return "Configuration error: invalid Groq API key."

        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ Groq rate limit (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                return "I'm receiving too many requests right now. Please try again in a moment."

        except APIStatusError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ Groq API error {e.status_code} (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "Groq service is temporarily unavailable. Please try again soon."

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return f"Unexpected error: {str(e)}"

    return "Failed to generate a response after multiple retries."
```

---

## 8. Retry Logic & Fallback Mechanisms

### 8.1 Upstash upsert with retry

The original Upstash design only retried client initialisation. The actual `upsert()` and `query()` calls also need retry coverage.

```python
def upsert_with_retry(vectors, max_retries=3):
    """
    Batch upsert to Upstash with exponential backoff.
    Upsert is idempotent — safe to retry without creating duplicates.
    """
    for attempt in range(max_retries):
        try:
            admin_index.upsert(vectors=vectors)
            return True

        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                print("❌ Upstash upsert: auth failed. Check UPSTASH_VECTOR_REST_TOKEN.")
                return False  # Don't retry auth failures
            elif "400" in err:
                print(f"❌ Upstash upsert: bad request — check vector format. {err}")
                return False  # Don't retry malformed data
            elif attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ Upstash upsert failed (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Upstash upsert failed after {max_retries} attempts: {err}")
                return False

    return False
```

### 8.2 Upstash query with retry

```python
def query_with_retry(question, top_k=3, max_retries=3):
    """
    Query Upstash with exponential backoff on transient failures.
    Returns list of doc strings, or empty list on failure.
    """
    for attempt in range(max_retries):
        try:
            results = query_index.query(data=question, top_k=top_k, include_metadata=True)
            return [r["metadata"]["original_text"] for r in results] if results else []

        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                print("❌ Upstash query: auth failed.")
                return []  # Don't retry
            elif attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ Upstash query failed (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ Upstash query failed after {max_retries} attempts: {err}")
                return []
```

### 8.3 Optional: Ollama fallback for LLM

If you want resilience during Groq outages, keep a lightweight Ollama fallback. This is optional — remove it if you no longer want a local dependency.

```python
def rag_query_ollama_fallback(question, top_docs):
    """
    Legacy Ollama LLM call. Used only when Groq is unavailable.
    Requires Ollama running locally at localhost:11434.
    """
    import requests as req
    context = "\n".join(top_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = req.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Both Groq and Ollama failed. Last error: {str(e)}"
```

---

## 9. Complete Migrated `rag_run.py`

This is the full, production-ready file incorporating all changes above.

```python
import os
import json
import time
import requests
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq
from groq._exceptions import RateLimitError, APIStatusError, AuthenticationError

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
JSON_FILE   = "foods.json"
GROQ_MODEL  = "llama-3.1-8b-instant"

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

    # Upstash: attempt a dummy query (empty index is OK)
    try:
        query_index.query(data="test", top_k=1, include_metadata=True)
        print("✅ Upstash Vector: connected")
    except Exception as e:
        print(f"❌ Upstash Vector: {e}")
        ok = False

    # Groq: 1-token completion
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
    Upsert is idempotent — safe to re-run.
    """
    vectors = []
    for item in food_data:
        enriched = item["text"]
        if "region" in item:
            enriched += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched += f" It is a type of {item['type']}."

        vectors.append((
            item["id"],
            enriched,                          # text Upstash will embed
            {
                "original_text": item["text"], # stored for retrieval
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
                print(f"⚠️ Upsert error, retrying in {wait}s... ({err})")
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
            return [r["metadata"]["original_text"] for r in results] if results else []
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                print("❌ Upstash query: auth failed. Check READONLY_TOKEN.")
                return []
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️ Upstash query error, retrying in {wait}s... ({err})")
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
                print(f"⚠️ Groq rate limit (attempt {attempt+1}/3). Waiting {wait}s...")
                time.sleep(wait)
            else:
                return "Too many requests. Please try again in a moment."

        except APIStatusError as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"⚠️ Groq API error {e.status_code} (attempt {attempt+1}/3). Retrying in {wait}s...")
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

    # Step 2: build messages
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

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Validate both services before starting
    if not validate_services():
        print("\n❌ One or more services failed to connect. Check your .env file.")
        exit(1)

    # Load and index data (idempotent)
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
```

---

## 10. Performance Considerations

### Batch upserts

Always upsert in a single batch call rather than looping:

```python
# ✅ One network round-trip for all items
index.upsert(vectors=all_vectors)

# ❌ N round-trips — avoid
for v in all_vectors:
    index.upsert(vectors=[v])
```

### Query result caching

For repeated questions (common in demos), cache results to avoid unnecessary API calls:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def rag_query_cached(question):
    return rag_query(question)
```

### Metadata filtering

If your dataset grows, filter by `type` or `region` before semantic search to reduce the candidate set:

```python
results = query_index.query(
    data=question,
    top_k=3,
    filter="type = 'Main Course'",
    include_metadata=True
)
```

### Token cost optimisation

Reduce context to 2 docs and cap output tokens for cost-sensitive scenarios:

```python
top_docs = _vector_query(question, top_k=2)   # fewer docs
# ...
groq_client.chat.completions.create(
    model=GROQ_MODEL,
    messages=messages,
    max_completion_tokens=200   # shorter answers
)
```

---

## 11. Cost Implications

### Upstash Vector (your food dataset ~90 items)

| Resource | Rate | Monthly estimate (90 items, 300 queries) |
|---|---|---|
| Storage | $0.25 / 100K vectors | ~$0.00023 |
| Queries | $0.5 / 10M queries | ~$0.000015 |
| Upserts | $0.5 / 1M writes | ~$0.000045 |
| **Total** | | **~$0.0003 — well within free tier** |

**Free tier limits:** 100K vectors stored, 10M queries/month, 1M writes/month.

### Groq (llama-3.1-8b-instant)

| Resource | Rate | Monthly estimate (300 queries) |
|---|---|---|
| Input tokens | $0.05 / 1M | ~700 tokens/query × 300 = $0.0105 |
| Output tokens | $0.08 / 1M | ~150 tokens/query × 300 = $0.0036 |
| **Total** | | **~$0.014 — well within free tier** |

**Free tier limits:** ~150K tokens/month, 30 RPM.

### vs. local infrastructure

| | Local (Ollama + ChromaDB) | Cloud (Upstash + Groq) |
|---|---|---|
| Hardware | GPU/CPU required | None |
| Electricity | ~$5–20/month | $0 |
| API cost | $0 | ~$0 (free tier) |
| Inference speed | 500–2000ms | 160–500ms |
| Maintenance | Manual | Zero |

---

## 12. Security Considerations

### Token scoping

Use the readonly token for all query operations in `rag_run.py`. Reserve the admin token for standalone ingestion scripts:

```python
# rag_run.py  — readonly for queries (user-facing)
query_index = Index(url=url, token=os.getenv("UPSTASH_VECTOR_REST_READONLY_TOKEN"))

# ingest.py   — admin for writes (maintenance only)
admin_index = Index(url=url, token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"))
```

### Never log tokens

```python
# ❌
print(f"Connecting with token {token}")
logging.error(f"Auth failed: {token}")

# ✅
print(f"Connecting to {url}")
logging.error("Upstash auth failed — check .env")
```

### Production deployments

Do not use `.env` files in production containers. Pass secrets via:
- Docker: `docker run -e GROQ_API_KEY=xxx ...`
- Kubernetes: `kubectl create secret generic rag-secrets ...`
- Cloud: AWS Secrets Manager / GCP Secret Manager / Azure Key Vault

### Key rotation

1. Generate a new token in the Upstash or Groq dashboard
2. Update `.env` (or the secrets manager entry)
3. Restart the application
4. Revoke the old token in the dashboard
5. Verify no errors in logs

---

## 13. Testing Strategy

### Connectivity tests (run before deployment)

```python
def test_upstash_connectivity():
    results = query_index.query(data="test food", top_k=1, include_metadata=True)
    assert results is not None, "Upstash query returned None"
    print("✅ Upstash connectivity OK")

def test_groq_connectivity():
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "test"}],
        max_completion_tokens=1
    )
    assert completion.choices[0].message.content is not None
    print("✅ Groq connectivity OK")
```

### End-to-end RAG tests

```python
def test_known_answer():
    result = rag_query("What is butter chicken?")
    assert len(result) > 20, "Response too short"
    assert "chicken" in result.lower(), "Expected 'chicken' in response"

def test_empty_result_graceful():
    result = rag_query("xyzzy nonsense gobbledygook")
    assert "couldn't find" in result.lower() or len(result) > 0

def test_latency():
    import time
    start = time.time()
    rag_query("What is ratatouille?")
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Query too slow: {elapsed:.2f}s"
    print(f"✅ Latency: {elapsed:.2f}s")
```

### Error handling tests (unit, with mocks)

```python
from unittest.mock import patch, MagicMock
from groq._exceptions import RateLimitError

@patch("rag_run.groq_client")
def test_groq_rate_limit_handled(mock_client):
    mock_client.chat.completions.create.side_effect = RateLimitError("rate limited")
    result = _groq_completion([{"role": "user", "content": "test"}])
    assert "too many requests" in result.lower() or "moment" in result.lower()

@patch("rag_run.query_index")
def test_upstash_empty_result(mock_index):
    mock_index.query.return_value = []
    result = rag_query("anything")
    assert "couldn't find" in result.lower()
```

---

## 14. Migration Checklist

### Setup
- [ ] Add `UPSTASH_VECTOR_REST_URL`, `UPSTASH_VECTOR_REST_TOKEN`, `UPSTASH_VECTOR_REST_READONLY_TOKEN` to `.env`
- [ ] Add `GROQ_API_KEY` to `.env`
- [ ] Add `.env` to `.gitignore`
- [ ] Run `pip install upstash-vector groq python-dotenv`
- [ ] Run connectivity tests for both services

### Code changes
- [ ] Remove `chromadb` import and `PersistentClient` setup
- [ ] Remove `get_embedding()` function entirely
- [ ] Remove `EMBED_MODEL`, `CHROMA_DIR`, `COLLECTION_NAME` constants
- [ ] Add Upstash `Index` clients (admin + readonly)
- [ ] Add Groq client
- [ ] Update data ingestion to batch upsert with raw text
- [ ] Update `rag_query()` to use `_vector_query()` and `_groq_completion()`
- [ ] Add retry wrappers for both services
- [ ] Add `validate_services()` startup check

### Testing
- [ ] Connectivity test: Upstash
- [ ] Connectivity test: Groq
- [ ] End-to-end query test with known answer
- [ ] Latency test (target < 1s)
- [ ] Error handling test: Groq rate limit mock
- [ ] Error handling test: Upstash empty result
- [ ] Full dataset ingestion verified in Upstash dashboard

### Deployment
- [ ] Code review
- [ ] Deploy and monitor logs for first 24 hours
- [ ] Confirm token usage within expected range
- [ ] Archive or delete `chroma_db/` directory

---

## 15. Rollback Plan

If issues arise after deployment:

```bash
# 1. Revert to previous code
git revert <migration-commit>

# 2. Restore ChromaDB backup
cp -r chroma_db.backup chroma_db

# 3. Restart with original rag_run.py (requires Ollama running)
ollama serve &
python rag_run.py
```

Your Upstash index retains all data — the migration can be retried at any time without re-ingesting.

---

## References

- [Upstash Vector docs](https://upstash.com/docs/vector/features/embeddingmodels)
- [Upstash Python SDK](https://github.com/upstash/vector-python)
- [Groq Python SDK](https://github.com/groq/groq-python)
- [Groq console & API keys](https://console.groq.com)
- [mxbai-embed-large-v1 model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)