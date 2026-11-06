from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────

persistent_directory = "db/chroma_db_retry"

# Embedding model
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_model = VoyageAIEmbeddings(model="voyage-4-lite")

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load existing vector database
db = Chroma(
    collection_name="company_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ──────────────────────────────────────────────────────────────────
# PYDANTIC MODEL FOR STRUCTURED OUTPUT
# ──────────────────────────────────────────────────────────────────

class QueryVariations(BaseModel):
    queries: List[str]


# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"

print(f"Original Query: {original_query}\n")


# ──────────────────────────────────────────────────────────────────
# STEP 1: GENERATE MULTIPLE QUERY VARIATIONS
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(
    QueryVariations,
    method="json_mode"
)

prompt = f"""
Generate exactly 3 different variations of the following query
that would help retrieve relevant documents.

Original query: {original_query}

Return the result in JSON format.

Your response must follow this JSON structure:

{{
    "queries": [
        "First query variation",
        "Second query variation",
        "Third query variation"
    ]
}}

Return only valid JSON.
"""

try:
    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    if len(query_variations) != 3:
        raise ValueError(
            f"Expected 3 query variations, but received {len(query_variations)}."
        )

except Exception as e:
    print("Error generating query variations:")
    print(e)
    raise SystemExit

print("Generated Query Variations:")

for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────
# STEP 2: SEARCH WITH EACH QUERY VARIATION
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

all_retrieval_results = []

for i, query in enumerate(query_variations, 1):

    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)

    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):

        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)


print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")
print("Different query variations may retrieve different documents.")


# ──────────────────────────────────────────────────────────────────
# STEP 3: RECIPROCAL RANK FUSION (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}

    chunk_id_map = {}
    chunk_counter = 1

    for query_idx, chunks in enumerate(chunk_lists, 1):

        if verbose:
            print(f"Processing Query {query_idx} results:")

        for position, chunk in enumerate(chunks, 1):

            # Identify a chunk using its content and metadata
            chunk_content = chunk.page_content
            chunk_metadata = str(sorted(chunk.metadata.items()))

            unique_key = (chunk_content, chunk_metadata)

            # Assign an ID if this chunk has not been seen before
            if unique_key not in chunk_id_map:

                chunk_id_map[unique_key] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[unique_key]

            all_unique_chunks[unique_key] = chunk

            # Calculate RRF score
            position_score = 1 / (k + position)

            rrf_scores[unique_key] += position_score

            if verbose:

                print(
                    f"  Position {position}: {chunk_id} "
                    f"+{position_score:.4f} "
                    f"(running total: {rrf_scores[unique_key]:.4f})"
                )

                print(f"    Preview: {chunk_content[:80]}...")

        if verbose:
            print()

    # Sort chunks by RRF score
    sorted_chunks = sorted(
        [
            (all_unique_chunks[unique_key], score)
            for unique_key, score in rrf_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )

    if verbose:
        print(
            f"RRF Complete! Processed {len(sorted_chunks)} "
            f"unique chunks from {len(chunk_lists)} queries."
        )

    return sorted_chunks


# Apply RRF
fused_results = reciprocal_rank_fusion(
    all_retrieval_results,
    k=60,
    verbose=True
)


# ──────────────────────────────────────────────────────────────────
# STEP 4: DISPLAY FINAL FUSED RESULTS
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(
    f"\nTop {min(10, len(fused_results))} "
    "documents after RRF fusion:\n"
)

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):

    print(f"RANK {rank} (RRF Score: {rrf_score:.4f})")

    print(f"Source: {doc.metadata.get('source', 'Unknown')}")

    print(f"{doc.page_content[:200]}...")

    print("-" * 50)


print(
    f"\nRRF Complete! Fused {len(fused_results)} unique documents "
    f"from {len(query_variations)} query variations."
)
print("\n" + "=" * 60)
print("RECIPROCAL RANK FUSION COMPLETED")
print("=" * 60)