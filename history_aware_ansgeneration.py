from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db_retry"
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_model = VoyageAIEmbeddings(model="voyage-4-lite")
db = Chroma(
    collection_name="company_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Set up AI model
# model = ChatOpenAI(model="gpt-4o")
model = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Include recent conversation context in the retrieval query
    if chat_history:
        previous_questions = [
            message.content
            for message in chat_history
            if isinstance(message, HumanMessage)
        ]

        search_question = " ".join(previous_questions[-3:] + [user_question])
    else:
        search_question = user_question

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 2: Prepare context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
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
    - If the documents do not contain enough information, say:
      "I don't have enough information to answer that question
      based on the provided documents."
    - Provide a clear and helpful answer.
    """

    # Step 3: Generate answer (ONE Groq API call)
    messages = [
        SystemMessage(
            content="You are a helpful RAG assistant. Answer questions using the provided documents and conversation history."
        ),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 4: Save conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")

    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()