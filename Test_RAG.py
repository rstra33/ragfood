"""
RAG-Food Comprehensive Query Test Suite
=======================================
Runs 15 structured test queries across 5 categories, measures response time,
logs token usage, and scores relevance. Results are saved to test_results.json
and printed as a formatted report.

Usage:
    python test_rag.py

Requirements:
    - .env with UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_READONLY_TOKEN,
      GROQ_API_KEY populated
    - pip install upstash-vector groq python-dotenv openpyxl
"""

import os
import json
import time
import statistics
from datetime import datetime
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq
from groq._exceptions import RateLimitError, APIStatusError

load_dotenv()

# ─── Clients ──────────────────────────────────────────────────────────────────
query_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_READONLY_TOKEN"),
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── Test definitions ─────────────────────────────────────────────────────────
# Each test has:
#   query        : the question sent to the RAG system
#   category     : one of 5 test categories
#   expected_ids : IDs of foods.json entries that SHOULD appear as sources
#   keywords     : words that a good answer must contain at least one of
#   description  : human-readable intent of the test

TEST_QUERIES = [

    # ── Category 1: Semantic Similarity ───────────────────────────────────────
    {
        "id": "T01",
        "category": "Semantic Similarity",
        "query": "healthy Mediterranean options",
        "expected_ids": ["66", "67", "69", "93", "95", "103"],
        "keywords": ["mediterranean", "hummus", "falafel", "greek", "tabbouleh",
                     "moussaka", "olive", "levantine"],
        "description": "Broad semantic search — no exact dish name given"
    },
    {
        "id": "T02",
        "category": "Semantic Similarity",
        "query": "light refreshing dishes with citrus",
        "expected_ids": ["55", "57", "58", "92", "103"],
        "keywords": ["citrus", "lime", "lemon", "fresh", "raw", "ceviche",
                     "poke", "kokoda", "oka"],
        "description": "Flavour-profile semantic search"
    },
    {
        "id": "T03",
        "category": "Semantic Similarity",
        "query": "warming winter soups and stews",
        "expected_ids": ["76", "77", "95", "96", "101", "107"],
        "keywords": ["stew", "soup", "braise", "slow", "warm", "broth",
                     "goulash", "borscht", "lentil", "tagine"],
        "description": "Context/season semantic search"
    },

    # ── Category 2: Multi-Criteria Searches ───────────────────────────────────
    {
        "id": "T04",
        "category": "Multi-Criteria",
        "query": "spicy vegetarian Asian dishes",
        "expected_ids": ["9", "25", "35", "41", "43", "94"],
        "keywords": ["spicy", "vegetarian", "vegan", "chili", "tofu",
                     "kimchi", "tteokbokki", "mapo", "chole"],
        "description": "Three simultaneous criteria: heat + diet + region"
    },
    {
        "id": "T05",
        "category": "Multi-Criteria",
        "query": "gluten-free dishes high in protein",
        "expected_ids": ["83", "84", "92", "97", "99", "104"],
        "keywords": ["gluten-free", "protein", "salmon", "almonds", "teff",
                     "edamame", "ceviche", "injera"],
        "description": "Dietary restriction + nutritional requirement"
    },
    {
        "id": "T06",
        "category": "Multi-Criteria",
        "query": "vegan dishes from Africa or the Middle East",
        "expected_ids": ["66", "67", "69", "90", "108"],
        "keywords": ["vegan", "africa", "middle east", "hummus", "falafel",
                     "jollof", "shakshuka", "tabbouleh"],
        "description": "Diet type + two geographic regions"
    },

    # ── Category 3: Nutritional Queries ───────────────────────────────────────
    {
        "id": "T07",
        "category": "Nutritional",
        "query": "high protein low carb foods",
        "expected_ids": ["12", "83", "84", "92", "104"],
        "keywords": ["protein", "low carb", "salmon", "almonds", "edamame",
                     "ceviche", "chicken", "omega"],
        "description": "Macronutrient-specific query"
    },
    {
        "id": "T08",
        "category": "Nutritional",
        "query": "foods rich in antioxidants and vitamins",
        "expected_ids": ["81", "82", "85", "99", "100", "102"],
        "keywords": ["antioxidant", "vitamin", "blueberr", "broccoli",
                     "spinach", "kale", "acai", "anthocyanin"],
        "description": "Micronutrient-focused query"
    },
    {
        "id": "T09",
        "category": "Nutritional",
        "query": "what should I eat for gut health and digestion",
        "expected_ids": ["41", "64", "99", "100", "101", "103", "104"],
        "keywords": ["probiotic", "ferment", "fibre", "fiber", "gut",
                     "digestion", "kimchi", "miso", "lentil", "edamame"],
        "description": "Health goal query — no dish name mentioned"
    },
    {
        "id": "T10",
        "category": "Nutritional",
        "query": "low calorie filling meals under 300 calories",
        "expected_ids": ["92", "96", "103", "108"],
        "keywords": ["calorie", "low", "light", "filling", "ceviche",
                     "borscht", "shakshuka", "fattoush", "soup"],
        "description": "Calorie-specific constraint query"
    },

    # ── Category 4: Cultural Exploration ──────────────────────────────────────
    {
        "id": "T11",
        "category": "Cultural Exploration",
        "query": "traditional comfort foods with interesting cultural stories",
        "expected_ids": ["105", "106", "107", "108", "109", "110"],
        "keywords": ["comfort", "tradition", "culture", "history", "heritage",
                     "pierogi", "goulash", "shakshuka", "shepherd",
                     "khachapuri", "mac"],
        "description": "Cultural + emotional category query"
    },
    {
        "id": "T12",
        "category": "Cultural Exploration",
        "query": "dishes that represent national identity or are UNESCO recognised",
        "expected_ids": ["86", "92", "96", "97", "107", "110"],
        "keywords": ["national", "unesco", "symbol", "identity", "heritage",
                     "paella", "borscht", "injera", "khachapuri"],
        "description": "Geopolitical/cultural significance query"
    },
    {
        "id": "T13",
        "category": "Cultural Exploration",
        "query": "fusion foods that blend two different culinary traditions",
        "expected_ids": ["94", "88", "31"],
        "keywords": ["fusion", "influence", "blend", "origin", "chifa",
                     "lomo saltado", "adobo", "chilaquiles", "chinese",
                     "spanish", "peruvian"],
        "description": "Abstract culinary concept query"
    },

    # ── Category 5: Cooking Method Queries ────────────────────────────────────
    {
        "id": "T14",
        "category": "Cooking Method",
        "query": "dishes that are slow cooked or braised for several hours",
        "expected_ids": ["76", "77", "89", "95", "107", "109"],
        "keywords": ["slow", "braise", "hours", "simmer", "tender",
                     "bourguignon", "tagine", "goulash", "ribs",
                     "shepherd", "coq au vin"],
        "description": "Technique-specific query"
    },
    {
        "id": "T15",
        "category": "Cooking Method",
        "query": "dishes cooked on a grill or open flame",
        "expected_ids": ["12", "33", "47", "53", "56", "89"],
        "keywords": ["grill", "flame", "barbecue", "roast", "charcoal",
                     "tandoor", "smoked", "peking", "hangi", "lovo",
                     "char siu", "tandoori"],
        "description": "Cooking method query with multiple valid answers"
    },
]

# ─── RAG pipeline (mirrors rag_run.py) ────────────────────────────────────────
def run_query(question):
    """Run the full RAG pipeline and return timing + result data."""

    # --- Vector search ---
    t0 = time.perf_counter()
    results = query_index.query(data=question, top_k=3, include_metadata=True)
    vector_ms = (time.perf_counter() - t0) * 1000

    top_docs = [r.metadata["original_text"] for r in results]
    top_ids  = [r.id for r in results]
    scores   = [round(r.score, 4) for r in results]

    if not top_docs:
        return None

    context = "\n".join(top_docs)
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

    # --- LLM call ---
    t1 = time.perf_counter()
    for attempt in range(3):
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=512,
                top_p=0.95
            )
            break
        except RateLimitError:
            if attempt < 2:
                print(f"    ⚠️  Rate limit, waiting {2**attempt}s...")
                time.sleep(2 ** attempt)
            else:
                raise
        except APIStatusError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

    llm_ms    = (time.perf_counter() - t1) * 1000
    total_ms  = (time.perf_counter() - t0) * 1000

    answer       = completion.choices[0].message.content.strip()
    input_tokens = completion.usage.prompt_tokens
    output_tokens= completion.usage.completion_tokens

    return {
        "answer":        answer,
        "retrieved_ids": top_ids,
        "scores":        scores,
        "vector_ms":     round(vector_ms, 1),
        "llm_ms":        round(llm_ms, 1),
        "total_ms":      round(total_ms, 1),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }

# ─── Scoring ──────────────────────────────────────────────────────────────────
def score_result(test, result):
    """
    Returns a relevance score 0-10:
      - Source overlap (0-4 pts): how many expected IDs appeared in top-3
      - Keyword match  (0-4 pts): how many expected keywords in the answer
      - Answer length  (0-2 pts): non-empty and substantive (>30 words)
    """
    if result is None:
        return 0, "No result returned"

    answer_lower = result["answer"].lower()

    # Source overlap
    retrieved = set(result["retrieved_ids"])
    expected  = set(test["expected_ids"])
    overlap   = len(retrieved & expected)
    src_score = min(overlap * 1.5, 4)  # 0, 1.5, 3, 4

    # Keyword match
    hits = sum(1 for kw in test["keywords"] if kw.lower() in answer_lower)
    kw_score = min(hits, 4)

    # Answer quality
    word_count = len(result["answer"].split())
    qual_score = 2 if word_count >= 30 else (1 if word_count >= 10 else 0)

    total = round(src_score + kw_score + qual_score, 1)
    detail = (f"src_overlap={overlap}/{len(expected)} "
              f"kw_hits={hits}/{len(test['keywords'])} "
              f"words={word_count}")
    return min(total, 10), detail

# ─── Main runner ──────────────────────────────────────────────────────────────
def run_all_tests():
    print("=" * 70)
    print("RAG-FOOD COMPREHENSIVE QUERY TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model:   {GROQ_MODEL}")
    print("=" * 70)

    all_results = []
    category_stats = {}

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[{test['id']}] {test['category']}")
        print(f"  Query: \"{test['query']}\"")

        try:
            result = run_query(test["query"])
            score, detail = score_result(test, result)

            print(f"  Sources: {result['retrieved_ids']} (scores: {result['scores']})")
            print(f"  Timing:  vector={result['vector_ms']}ms | "
                  f"llm={result['llm_ms']}ms | total={result['total_ms']}ms")
            print(f"  Tokens:  {result['input_tokens']} in / {result['output_tokens']} out")
            print(f"  Score:   {score}/10  ({detail})")
            print(f"  Answer:  {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")

            row = {
                "test_id":        test["id"],
                "category":       test["category"],
                "query":          test["query"],
                "description":    test["description"],
                "score":          score,
                "score_detail":   detail,
                "retrieved_ids":  result["retrieved_ids"],
                "answer":         result["answer"],
                "vector_ms":      result["vector_ms"],
                "llm_ms":         result["llm_ms"],
                "total_ms":       result["total_ms"],
                "input_tokens":   result["input_tokens"],
                "output_tokens":  result["output_tokens"],
                "status":         "pass" if score >= 5 else "fail"
            }

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            row = {
                "test_id":      test["id"],
                "category":     test["category"],
                "query":        test["query"],
                "description":  test["description"],
                "score":        0,
                "score_detail": str(e),
                "status":       "error"
            }

        all_results.append(row)

        # Throttle to avoid Groq free-tier rate limits (30 RPM)
        if i < len(TEST_QUERIES):
            time.sleep(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    categories = list(dict.fromkeys(t["category"] for t in TEST_QUERIES))
    for cat in categories:
        cat_rows = [r for r in all_results if r["category"] == cat
                    and "score" in r and isinstance(r.get("score"), (int, float))]
        if cat_rows:
            scores   = [r["score"] for r in cat_rows]
            timings  = [r["total_ms"] for r in cat_rows if "total_ms" in r]
            avg_score = round(statistics.mean(scores), 1)
            avg_time  = round(statistics.mean(timings), 0) if timings else "N/A"
            passes    = sum(1 for r in cat_rows if r.get("status") == "pass")
            print(f"\n  {cat}")
            print(f"    Avg score : {avg_score}/10  |  Pass rate: {passes}/{len(cat_rows)}")
            print(f"    Avg time  : {avg_time}ms")

    scored = [r for r in all_results if isinstance(r.get("score"), (int, float))]
    timed  = [r for r in all_results if "total_ms" in r]

    if scored:
        all_scores = [r["score"] for r in scored]
        all_times  = [r["total_ms"] for r in timed]
        total_in   = sum(r.get("input_tokens", 0) for r in all_results)
        total_out  = sum(r.get("output_tokens", 0) for r in all_results)

        print(f"\n  Overall")
        print(f"    Avg score  : {round(statistics.mean(all_scores), 1)}/10")
        print(f"    Min/Max    : {min(all_scores)}/{max(all_scores)}")
        print(f"    Avg latency: {round(statistics.mean(all_times), 0)}ms")
        print(f"    Min/Max lat: {min(all_times)}ms / {max(all_times)}ms")
        print(f"    Total tokens: {total_in} in / {total_out} out")
        est_cost = (total_in / 1_000_000 * 0.05) + (total_out / 1_000_000 * 0.08)
        print(f"    Est. cost  : ${est_cost:.5f}")
        passes = sum(1 for r in all_results if r.get("status") == "pass")
        print(f"    Pass (≥5)  : {passes}/{len(TEST_QUERIES)}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "model": GROQ_MODEL,
        "total_tests": len(TEST_QUERIES),
        "results": all_results
    }

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Full results saved to test_results.json")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_all_tests()