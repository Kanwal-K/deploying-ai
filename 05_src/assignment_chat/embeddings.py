import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("05_src/.secrets")

# --- Your course client ---
client = OpenAI(
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# --- Load text ---
with open("05_src/assignment_chat/data/rag_slides.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Text loaded successfully")

# --- Chunk the text ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.create_documents([text])
print(f"Number of chunks: {len(chunks)}")

# --- Generate embeddings manually ---
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# --- Set up ChromaDB ---
chroma_client = chromadb.PersistentClient(path="05_src/assignment_chat/chroma_db")

collection = chroma_client.get_or_create_collection(name="course_materials")

# --- Add chunks with embeddings ---
for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk.page_content)
    collection.add(
        documents=[chunk.page_content],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )
    print(f"Added chunk {i+1}/{len(chunks)}")

print("Done! ChromaDB saved successfully")