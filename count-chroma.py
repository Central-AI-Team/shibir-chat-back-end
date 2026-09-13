import chromadb

client = chromadb.PersistentClient(path="/app/chroma_db")
collections = client.list_collections()
print(f"Total collections: {len(collections)}\n")
for c in collections:
    print(f"- {c.name}: {c.count()} items")
