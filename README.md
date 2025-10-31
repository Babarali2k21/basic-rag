# Basic RAG (Retrieval-Augmented Generation)

## Overview  
This project is a simple **Python implementation of Retrieval-Augmented Generation (RAG)**.  
It demonstrates how to:  
- Load and split `.txt` documents  
- Generate embeddings using **OpenAI**  
- Store and query embeddings in **ChromaDB**  
- Generate concise answers with context retrieved from documents  

It was created to practice **document retrieval, vector databases, and LLM-based question answering**.  

---

## Tools & Libraries  
The project uses the following Python libraries:  
- **openai** → to generate embeddings and chat responses  
- **chromadb** → as the vector database for storing and retrieving document chunks  
- **python-dotenv** → to manage API keys securely in a `.env` file  

---

## Files in this Repository  
- `main.py` → Entry point to run the pipeline and ask questions interactively  
- `config.py` → Handles environment variables and configuration  
- `data_loader.py` → Loads documents and splits them into chunks  
- `embeddings.py` → Generates embeddings using OpenAI  
- `vector_store.py` → Sets up and manages ChromaDB collection  
- `rag.py` → RAG pipeline logic (retrieval + response generation)  
- `articles/` → Folder containing `.txt` files used as knowledge base  
- `requirements.txt` → List of dependencies  

---

## Workflow  
1. Load documents → split into chunks  
2. Create embeddings for each chunk  
3. Store embeddings in ChromaDB  
4. User asks a question  
5. Retrieve top matching chunks  
6. LLM generates an answer using context 

![Coverage](coverage.svg)