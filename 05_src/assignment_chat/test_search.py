import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("05_src/.secrets")

client = OpenAI(
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# --- Load ChromaDB ---
chroma_client = chromadb.PersistentClient(path="05_src/assignment_chat/chroma_db")
collection = chroma_client.get_collection(name="course_materials")

# --- Test query ---
def search(query, n_results=3):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]

# --- Test it ---
query = "what is RAG?"
results = search(query)

print(f"Query: {query}")
print()
for i, result in enumerate(results):
    print(f"Result {i+1}: {result}")
    print()