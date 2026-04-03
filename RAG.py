# rag_chatbot.py
# Simple RAG — Ask questions from any .txt file
# Install: pip install langchain langchain-community chromadb sentence-transformers

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ── STEP 1: Load your document ─────────────────────────────
loader = TextLoader("knowledge.txt")       # your text file
documents = loader.load()

# ── STEP 2: Split into chunks ──────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ── STEP 3: Create embeddings + store in ChromaDB ──────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── STEP 4: Load a free LLM ────────────────────────────────
pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# ── STEP 5: Build RAG chain ────────────────────────────────
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ── STEP 6: Ask questions ──────────────────────────────────
while True:
    query = input("\nAsk a question (or 'quit'): ")
    if query.lower() == "quit":
        break
    answer = rag_chain.invoke(query)
    print(f"\nAnswer: {answer['result']}")
```


---

## How to put it on GitHub — step by step
```
# In terminal:
mkdir rag-qa-chatbot
cd rag-qa-chatbot
git init

# Create these files:
# rag_chatbot.py  ← paste the code above
# knowledge.txt   ← any text content you want to query
# requirements.txt ← paste the 4 lines below
```

`requirements.txt`:
```
langchain
langchain-community
chromadb
sentence-transformers
transformers
