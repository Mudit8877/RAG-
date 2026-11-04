import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
#load documents
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True
        }
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    # for i, doc in enumerate(documents[:2]):  # Show first 2 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")

    return documents

#Splitter
def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # if chunks:
    
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
        
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks
#vector store and embeddings using openai model
# def create_vector_store(chunks, persist_directory="db/chroma_db"):
#     """Create and persist ChromaDB vector store"""
#     print("Creating embeddings and storing in ChromaDB...")
        
#     # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
#     # Create ChromaDB vector store
#     print("--- Creating vector store ---")
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=persist_directory, 
#         collection_metadata={"hnsw:space": "cosine"}
#     )
#     print("--- Finished creating vector store ---")
    
#     print(f"Vector store created and saved to {persist_directory}")
#     return vectorstore

#Method 2 with reduced rate limits with free voyageAI(With lite model takes about 5 Hour to get the Vector Database)
def create_vector_store(chunks, persist_directory="db/chroma_db_retry"):
    """Create and persist ChromaDB vector store"""

    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = VoyageAIEmbeddings(
        model="voyage-4-lite"
    )

    print("--- Creating vector store ---")

    # Create an empty Chroma collection
    vectorstore = Chroma(
        collection_name="company_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Process chunks in small batches
    batch_size = 2

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        print(f"Processing chunks {i + 1} to {i + len(batch)}")

        vectorstore.add_documents(batch)

        print("Batch successfully stored!")

        # Wait between requests
        if i + batch_size < len(chunks):
            time.sleep(22)

    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    print("Main Function")
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
