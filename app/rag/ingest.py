import sqlite3
import chromadb
from bs4 import BeautifulSoup
from app.rag.embedder import embed_text

# Connect to SQLite database
conn = sqlite3.connect('Tarun_Associate.db')
cursor = conn.cursor()

# Connect to ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()

def ingest_data():
    print("Fetching data from database...")
    # Join pages with books and chapters to get context
    query = """
        SELECT 
            p.id, 
            b.name as book_name, 
            c.name as chapter_name, 
            p.content 
        FROM pages p
        LEFT JOIN books b ON p.book = b.id
        LEFT JOIN chapters c ON p.chapter = c.id
        WHERE p.content IS NOT NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"Found {len(rows)} pages. Starting ingestion...")

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for row in rows:
        page_id, book_name, chapter_name, content = row
        
        # Clean HTML content
        clean_text = clean_html(content)
        
        if not clean_text:
            continue

        # Create a descriptive chunk
        full_text = f"Book: {book_name}\nChapter: {chapter_name}\nContent:\n{clean_text}"
        
        # Generate embedding
        embedding = embed_text(full_text)

        ids.append(str(page_id))
        documents.append(full_text)
        metadatas.append({
            "book": book_name or "Unknown",
            "chapter": chapter_name or "Unknown",
            "page_id": page_id
        })
        embeddings.append(embedding)

        # Batch insert every 100 items
        if len(ids) >= 100:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"Ingested batch ending at page {page_id}")
            ids = []
            documents = []
            metadatas = []
            embeddings = []

    # Insert remaining
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
    conn.close()
