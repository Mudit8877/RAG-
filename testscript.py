#Details about database to check the chunks and the Files
import chromadb

client = chromadb.PersistentClient(
    path="db/chroma_db_retry"
)

collection = client.get_collection(
    name="company_documents"
)

data = collection.get(
    include=["metadatas"]
)

print("Total existing chunks:", len(data["ids"]))

print("\nExisting document sources:")

sources = set()

for metadata in data["metadatas"]:
    if metadata and "source" in metadata:
        sources.add(metadata["source"])

for source in sorted(sources):
    print(source)