import os
from dotenv import load_dotenv
from langchain.embeddings import AzureOpenAIEmbeddings
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the embeddings model

emb = AzureOpenAIEmbeddings()

# Load a PDF document and split into chunks
# See https://github.com/ollama/ollama/blob/main/examples/langchain-python-rag-document/main.py
loader = PyPDFLoader("Sample Report.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = loader.load()
doc_chunks = splitter.split_documents(docs)

# Print the first and last chunks
print(f"Loaded document ({len(doc_chunks)} chunks):\n")
print(f"{doc_chunks[0].page_content}\n\n...\n\n{doc_chunks[-1].page_content}")