import os
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import AzureChatOpenAI
import CheckEmbedding
# Load the document chunks into a vector database
db = Chroma.from_documents(documents=CheckEmbedding.doc_chunks, embedding=CheckEmbedding.emb)
retriever = db.as_retriever()

# Set up a chain for RAG question/answers
sys_prompt = """
You are an assistant for question-answering tasks. Use the following \
pieces of retrieved context to answer the question. If you don't know \
the answer, just say that you don't know. Use three sentences \
maximum and keep the answer concise.

Context: {context}
"""
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', sys_prompt),
    ('human', "{input}")
])

# Set up a chat model
rag_model = AzureChatOpenAI(
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature = 0.8,
    max_tokens = 60
)
# Create the RAG chain
docs_chain = create_stuff_documents_chain(rag_model, rag_prompt)
chain = create_retrieval_chain(retriever, docs_chain)

# Retrieves an answer
def query(question: str) -> dict:
    res = chain.invoke({
        'input': question,
    })
    return res

# Ask a question based on the document
res = query("Explain what this pdf is about in a sentence")
print(res['answer'])