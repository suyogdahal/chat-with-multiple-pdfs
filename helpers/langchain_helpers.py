from typing import List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger


def get_vectorstore(documents: List[Document]):
    embeddings = OpenAIEmbeddings(show_progress_bar=True)
    logger.info("Creating Embeddings")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    logger.info("Sucessfully created a vectorstore")
    return vectorstore


def get_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o")
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
