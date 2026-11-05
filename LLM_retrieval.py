
import os
import json
import time
from datetime import date

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db_retry"

# ----------------------------------
# 1. Groq daily request limiter
# ----------------------------------

DAILY_REQUEST_LIMIT = 22
USAGE_FILE = "groq_usage.json"

today = str(date.today())

if os.path.exists(USAGE_FILE):
    with open(USAGE_FILE, "r") as file:
        usage_data = json.load(file)
else:
    usage_data = {"date": today, "requests": 0}

# Reset counter when the date changes
if usage_data["date"] != today:
    usage_data = {"date": today, "requests": 0}

# Check daily limit before making an API request
if usage_data["requests"] >= DAILY_REQUEST_LIMIT:
    print("Daily Groq request limit reached.")
    print(f"Requests used: {usage_data['requests']}/{DAILY_REQUEST_LIMIT}")
    print("Please try again after the counter resets.")
    raise SystemExit

print(
    f"Groq requests remaining: "
    f"{DAILY_REQUEST_LIMIT - usage_data['requests']}"
)

# ----------------------------------
# 2. Load embeddings and vector store
# ----------------------------------
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

embedding_model = VoyageAIEmbeddings(
    model="voyage-4-lite"
)

db = Chroma(
    collection_name="company_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ----------------------------------
# 3. Search for relevant documents
# ----------------------------------

query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# ----------------------------------
# 4. Combine query and context
# ----------------------------------

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# ----------------------------------
# 5. Initialize Groq model
# ----------------------------------

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# ----------------------------------
# 6. Invoke Groq with retry logic
# ----------------------------------

max_retries = 3
result = None

for attempt in range(max_retries):
    try:
        # Count the request before sending it
        usage_data["requests"] += 1

        with open(USAGE_FILE, "w") as file:
            json.dump(usage_data, file, indent=4)

        result = model.invoke(messages)
        break

    except Exception as e:
        error_message = str(e)

        # Stop if API rate limit is exceeded
        if "429" in error_message or "rate_limit" in error_message.lower():
            print("\nGroq API rate limit exceeded.")
            print("Please check your Groq Console usage.")
            break

        # Retry temporary server errors
        elif "503" in error_message or "UNAVAILABLE" in error_message:
            if attempt < max_retries - 1:
                print(
                    "Groq temporarily unavailable. "
                    "Retrying in 10 seconds..."
                )
                time.sleep(10)
            else:
                print("Groq is still unavailable. Please try again later.")

        else:
            print("\nUnexpected error occurred:")
            print(e)
            break

# ----------------------------------
# 7. Display generated response
# ----------------------------------

if result is not None:
    print("\n--- Generated Response ---")
    print("Content only:")
    print(result.content)
else:
    print("\nNo response was generated.")

# ----------------------------------
# 8. Display daily usage
# ----------------------------------

print(
    f"\nGroq requests used today: "
    f"{usage_data['requests']}/{DAILY_REQUEST_LIMIT}"
)
