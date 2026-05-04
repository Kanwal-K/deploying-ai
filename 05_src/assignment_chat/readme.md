# Professor Knowitall 🎓

A sarcastic and dramatic AI teaching assistant for the Deploying AI course at the University of Toronto. Professor Knowitall is mildly offended by easy questions but always helps in the end.

## How to Run

1. The `.secrets` file should be set up with the `API_GATEWAY_KEY`
2. The `chroma_db` folder is already included — no need to run `embeddings.py` again
3. Run the chatbot from the root of the repository:

```bash
python 05_src/assignment_chat/chat.py
```

4. Open your browser at `http://127.0.0.1:7860`

## Services

### Service 1: Wikipedia API
When a student asks a general AI or machine learning question, Professor Knowitall fetches a summary from the Wikipedia REST API and rephrases it in his own dramatic style. No API key is needed for this service.

### Service 2: Semantic Search
Course materials from the RAG lecture slides are stored as embeddings in a ChromaDB database. When a student asks a course-related question, the system converts the query into an embedding and retrieves the most relevant chunks using cosine similarity.

### Service 3: Function Calling
The model uses OpenAI function calling to decide which tool to use depending on what the student asks.

- `search_course_materials` — searches the ChromaDB embeddings
- `search_wikipedia` — fetches from Wikipedia API
- `get_random_quiz_question` — returns a random quiz question from the course material
- `explain_concept` — gives a one-line definition of a concept
- `suggest_resources` — suggests what to study for a given topic

## Embedding Process

Embeddings were generated from the RAG lecture slides (`data/rag_slides.txt`) using the `text-embedding-3-small` model from OpenAI. The text was split into chunks of 500 characters with 50 character overlap using LangChain's `RecursiveCharacterTextSplitter`. The resulting embeddings are stored in a persistent ChromaDB instance in the `chroma_db/` folder.

The code used to generate embeddings is in `embeddings.py` but does not need to be re-run as the database is already saved.

## Guardrails

Professor Knowitall includes the following guardrails:

- Refuses to reveal or modify the system prompt
- Refuses to discuss cats, dogs, horoscopes, zodiac signs, or Taylor Swift
- Only answers questions related to AI, machine learning, and data science

## Memory

The chat maintains conversation history through Gradio's built-in history parameter. To prevent context window overflow, only the last 10 messages are kept in memory.

## Files

```
assignment_chat/
├── chat.py          — main chatbot file
├── embeddings.py    — used to generate and save embeddings (already run)
├── test_search.py   — used during development to test chromadb search
├── data/
│   └── rag_slides.txt   — course slides used as knowledge base
└── chroma_db/           — saved chromadb embeddings
```