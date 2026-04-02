# RAG Q&A Chatbot using LangChain & ChromaDB

A simple Retrieval-Augmented Generation (RAG) system that lets you ask 
natural language questions over any text document.

## How it works
1. Loads a .txt document
2. Splits it into chunks and converts to vector embeddings (all-MiniLM-L6-v2)
3. Stores embeddings in ChromaDB vector store
4. On each query, retrieves the top-3 most relevant chunks
5. Passes chunks + question to Flan-T5 LLM to generate an answer

## Stack
- LangChain — orchestration
- ChromaDB — vector store
- Hugging Face (all-MiniLM-L6-v2) — embeddings
- Flan-T5-small — free open-source LLM

## Run it
pip install -r requirements.txt
python rag_chatbot.py
```

Then push:
```
git add .
git commit -m "Initial RAG chatbot implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rag-qa-chatbot.git
git push -u origin main
