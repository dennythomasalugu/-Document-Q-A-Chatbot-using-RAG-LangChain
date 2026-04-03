from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

loader = TextLoader("knowledge.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("\nAsk a question (or 'quit'): ")
    if query.lower() == "quit":
        break
    answer = rag_chain.invoke(query)
    print(f"\nAnswer: {answer['result']}")
