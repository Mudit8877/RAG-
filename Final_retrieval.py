
import os
import json
import time
from datetime import date
from collections import defaultdict
from typing import List

import voyageai
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)


load_dotenv()


# ==========================================
# 1. CONFIGURATION
# ==========================================

PERSISTENT_DIRECTORY = "db/chroma_db_retry"
COLLECTION_NAME = "company_documents"

EMBEDDING_MODEL_NAME = "voyage-4-lite"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

DAILY_REQUEST_LIMIT = 22
USAGE_FILE = "groq_usage.json"

RETRIEVAL_K = 3
FINAL_CONTEXT_K = 3
RRF_K = 60


# ==========================================
# 2. GROQ DAILY REQUEST LIMITER
# ==========================================

today = str(date.today())

if os.path.exists(USAGE_FILE):

    try:
        with open(USAGE_FILE, "r") as file:
            usage_data = json.load(file)

    except (json.JSONDecodeError, OSError):
        usage_data = {
            "date": today,
            "requests": 0
        }

else:
    usage_data = {
        "date": today,
        "requests": 0
    }


# Reset counter when date changes
if usage_data.get("date") != today:

    usage_data = {
        "date": today,
        "requests": 0
    }


def save_usage():

    with open(USAGE_FILE, "w") as file:
        json.dump(usage_data, file, indent=4)


def invoke_groq(callable_function):

    max_retries = 3

    for attempt in range(max_retries):

        if usage_data["requests"] >= DAILY_REQUEST_LIMIT:

            print("\nDaily Groq request limit reached.")

            print(
                f"Requests used: "
                f"{usage_data['requests']}/{DAILY_REQUEST_LIMIT}"
            )

            return None

        # Count each API attempt
        usage_data["requests"] += 1
        save_usage()

        try:

            return callable_function()

        except Exception as e:

            error_message = str(e)

            if (
                "429" in error_message
                or "rate_limit" in error_message.lower()
            ):

                print("\nGroq API rate limit exceeded.")
                print("Please check your Groq Console usage.")

                return None

            elif (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries - 1:

                    print(
                        "\nGroq temporarily unavailable. "
                        "Retrying in 10 seconds..."
                    )

                    time.sleep(10)

                else:

                    print(
                        "\nGroq is still unavailable. "
                        "Please try again later."
                    )

                    return None

            else:

                print("\nUnexpected Groq error:")
                print(e)

                return None

    return None


print(
    f"Groq requests remaining: "
    f"{max(0, DAILY_REQUEST_LIMIT - usage_data['requests'])}"
)


# ==========================================
# 3. LOAD EMBEDDING MODEL AND CHROMADB
# ==========================================

embedding_model = VoyageAIEmbeddings(
    model=EMBEDDING_MODEL_NAME
)

# Voyage AI client for batch query embedding
voyage_client = voyageai.Client()

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

print("\nChromaDB loaded successfully.")


# ==========================================
# 4. INITIALIZE GROQ MODEL
# ==========================================

model = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0
)


# ==========================================
# 5. CHAT HISTORY
# ==========================================

chat_history = []


# ==========================================
# 6. MULTI-QUERY GENERATION
# ==========================================

class QueryVariations(BaseModel):

    queries: List[str]


query_model = model.with_structured_output(
    QueryVariations,
    method="json_mode"
)


def generate_query_variations(question):

    history_text = "\n".join(
        [
            f"{message.type}: {message.content}"
            for message in chat_history[-6:]
        ]
    )

    prompt = f"""
You are a query generation assistant for a RAG system.

Generate exactly 3 alternative search queries
for the current user question.

Instructions:

1. Preserve the original meaning.
2. Use conversation history to understand context.
3. Resolve references such as it, they, its and their.
4. Explore different relevant search perspectives.
5. Make queries suitable for document retrieval.
6. Do not answer the question.
7. Return exactly 3 alternative queries.

Conversation History:

{history_text}

Current Question:

{question}

Return the result in JSON format using this structure:

{{
    "queries": [
        "Alternative query 1",
        "Alternative query 2",
        "Alternative query 3"
    ]
}}
"""

    response = invoke_groq(
        lambda: query_model.invoke(prompt)
    )

    if response is None:
        return []

    unique_queries = []

    for query in response.queries:

        query = query.strip()

        if (
            query
            and query.lower() not in [
                q.lower() for q in unique_queries
            ]
        ):

            unique_queries.append(query)

        if len(unique_queries) == 3:
            break

    return unique_queries


# ==========================================
# 7. RECIPROCAL RANK FUSION
# ==========================================

def reciprocal_rank_fusion(chunk_lists, k=60):

    rrf_scores = defaultdict(float)
    unique_chunks = {}

    for chunks in chunk_lists:

        seen_in_list = set()

        for rank, doc in enumerate(chunks, start=1):

            # Identify duplicate chunks
            chunk_id = (
                doc.page_content,
                doc.metadata.get("source", ""),
                str(doc.metadata.get("page", ""))
            )

            # Avoid duplicate counting in the same list
            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            unique_chunks[chunk_id] = doc

            # Calculate RRF score
            rrf_scores[chunk_id] += 1 / (k + rank)

    # Sort chunks by RRF score
    ranked_chunks = sorted(
        unique_chunks.items(),
        key=lambda item: rrf_scores[item[0]],
        reverse=True
    )

    return [
        (doc, rrf_scores[chunk_id])
        for chunk_id, doc in ranked_chunks
    ]


# ==========================================
# 8. HISTORY-AWARE MULTI-QUERY RETRIEVAL
# ==========================================

def retrieve_documents(user_question):

    print(f"\n--- You asked: {user_question} ---")

    # ----------------------------------
    # Step 1: Prepare history-aware query
    # ----------------------------------

    previous_questions = [
        message.content
        for message in chat_history
        if isinstance(message, HumanMessage)
    ]

    if previous_questions:

        search_question = " ".join(
            previous_questions[-3:] + [user_question]
        )

    else:

        search_question = user_question

    print("\n--- History-Aware Query ---")
    print(search_question)

    # ----------------------------------
    # Step 2: Generate 3 alternative queries
    # ----------------------------------

    query_variations = generate_query_variations(
        search_question
    )

    if len(query_variations) != 3:

        print(
            "\nCould not generate 3 alternative queries."
        )

        return []

    # Original query + 3 alternatives
    all_queries = [search_question] + query_variations

    print("\n--- Generated Search Queries ---")

    for i, query in enumerate(all_queries, 1):

        print(f"{i}. {query}")

    try:

        # ----------------------------------
        # Step 3: Batch query embedding
        # ----------------------------------

        print(
            "\nEmbedding all 4 queries "
            "in one Voyage AI request..."
        )

        embedding_response = voyage_client.embed(
            texts=all_queries,
            model=EMBEDDING_MODEL_NAME,
            input_type="query"
        )

        query_embeddings = embedding_response.embeddings

        if len(query_embeddings) != len(all_queries):

            print("\nUnexpected number of query embeddings.")
            return []

        print(
            f"Successfully embedded "
            f"{len(query_embeddings)} queries."
        )

        # ----------------------------------
        # Step 4: Similarity search
        # ----------------------------------

        all_retrieval_results = []

        for i, query_embedding in enumerate(
            query_embeddings, 1
        ):

            print(f"\nSearching using query {i}...")

            docs = db.similarity_search_by_vector(
                embedding=query_embedding,
                k=RETRIEVAL_K
            )

            print(f"Found {len(docs)} documents.")

            all_retrieval_results.append(docs)

        # ----------------------------------
        # Step 5: Apply RRF
        # ----------------------------------

        print("\nApplying Reciprocal Rank Fusion...")

        ranked_chunks = reciprocal_rank_fusion(
            all_retrieval_results,
            k=RRF_K
        )

        # Select top 3 documents
        final_documents = [
            doc
            for doc, score in ranked_chunks[:FINAL_CONTEXT_K]
        ]

        print("\n--- Final RRF Results ---")

        for i, (doc, score) in enumerate(
            ranked_chunks[:FINAL_CONTEXT_K], 1
        ):

            print(
                f"\nDocument {i} | "
                f"RRF Score: {score:.5f}"
            )

            print(
                f"Source: "
                f"{doc.metadata.get('source', 'Unknown')}"
            )

            print(doc.page_content)

        print(
            f"\nSelected {len(final_documents)} "
            f"final documents."
        )

        return final_documents

    except Exception as e:

        print("\nRetrieval error:")
        print(e)

        return []


# ==========================================
# 9. FINAL ANSWER GENERATION
# ==========================================

def generate_answer(user_question, documents):

    # Prepare retrieved context
    context = "\n\n".join(
        [
            f"Document {i}:\n{doc.page_content}"
            for i, doc in enumerate(documents, 1)
        ]
    )

    combined_input = f"""
Answer the user's question using the provided documents
and conversation history.

Previous conversation:

{chat_history}

Retrieved documents:

{context}

Current question:

{user_question}

Instructions:

- Consider the previous conversation when understanding the question.
- Use only the retrieved documents to answer factual questions.
- Do not treat previous assistant responses as verified facts.
- If the documents do not contain enough information, say:

  "I don't have enough information to answer that question
  based on the provided documents."

- Provide a clear and helpful answer.
"""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful RAG assistant. "
                "Answer questions using the provided documents "
                "and conversation history."
            )
        ),
        HumanMessage(content=combined_input)
    ]

    # One Groq API call for final answer
    result = invoke_groq(
        lambda: model.invoke(messages)
    )

    if result is None:
        return None

    return result.content


# ==========================================
# 10. ASK QUESTION
# ==========================================

def ask_question(user_question):

    # Step 1: Retrieve documents
    docs = retrieve_documents(user_question)

    if not docs:

        print("\nNo relevant documents were retrieved.")
        return None

    # Step 2: Display document previews
    print(f"\nFound {len(docs)} final documents:")

    for i, doc in enumerate(docs, 1):

        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)

        print(f"  Document {i}: {preview}...")

    # Step 3: Generate final answer
    answer = generate_answer(
        user_question,
        docs
    )

    if answer is None:

        print("\nNo response was generated.")
        return None

    # Step 4: Save conversation
    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    print("\n--- Generated Response ---")
    print(answer)

    print(
        f"\nGroq requests used today: "
        f"{usage_data['requests']}/{DAILY_REQUEST_LIMIT}"
    )

    return answer


# ==========================================
# 11. SIMPLE CHAT LOOP
# ==========================================

def start_chat():

    print("\n===================================")
    print("       RAG CHAT ASSISTANT")
    print("===================================")

    print("Type 'quit' to exit.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["quit", "exit"]:

            print("\nConversation ended.")
            break

        if not question:

            print("Please enter a question.")
            continue

        ask_question(question)


# ==========================================
# 12. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    start_chat()
