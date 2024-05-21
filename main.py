from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from langchain.chains import conversational_retrieval
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from PyPDF2 import PdfReader

EMBEDDINGS = OpenAIEmbeddings(show_progress_bar=True)
CLIENT = ChatOpenAI()


@dataclass
class PdfInfo:
    """A naive dataclass for storing pdf information"""

    doc_name: str
    page_info: Dict[int, str]  # dict with page number as key and the text as value

    @property
    def text(self):
        return " ".join(self.page_info.values())

    @property
    def as_documents(self):
        """breaks a pdffile into multiple langchain docs with proper metadata"""
        documents = []
        for pg_num, text in self.page_info.items():
            document = Document(
                page_content=text,
                metadata={"doc_name": self.doc_name, "page_num": pg_num},
            )
            documents.append(document)
        return documents

    @classmethod
    def from_file(cls, path: str):
        """Constructor"""
        pdf_reader = PdfReader(path)
        page_info = {}
        for pg_num, page in enumerate(pdf_reader.pages):
            _text = page.extract_text()
            page_info[pg_num] = _text

        return cls(doc_name=path, page_info=page_info)


def get_pdf_infos(pdf_docs: List[str]) -> List[PdfInfo]:
    pdf_infos = []
    for pdf in pdf_docs:
        pdf_info = PdfInfo.from_file(pdf)
        pdf_infos.append(pdf_info)
    return pdf_infos


def get_documents(pdf_infos: List[PdfInfo]) -> List[Document]:
    documents = []
    for pdf_info in pdf_infos:
        documents.extend(pdf_info.as_documents())
    return documents


def get_splitted_docs(documents: List[Document]):
    logger.info("Splitting the documents into chunks ...")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Splitted into {len(chunks)} chunks.")
    return chunks


def get_vectorstore(documents: List[Document]):
    logger.info("Creating Embeddings")
    vectorstore = FAISS.from_documents(documents=documents, embedding=EMBEDDINGS)
    logger.info("Sucessfully created a vectorstore")
    return vectorstore


def get_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = conversational_retrieval(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    logger.debug(f"output: {conversation_chain}")
    return conversation_chain


def st_file_handler():
    """Handles any uploaded files and updates st session's vectorstore and chain"""
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=".pdf",
        )
        if st.button("Process"):
            if not pdf_docs:
                st.error("No file(s) selected. Please upload your PDF documents.")
                return
            with st.spinner("Processing"):
                logger.info(f"Processing {pdf_docs}")
                pdf_infos = get_pdf_infos(pdf_docs)
                documents = get_documents(pdf_infos)
                documents_chunks = get_splitted_docs(documents=documents)

                if st.session_state.vectorstore is None:
                    logger.debug("No vectorstore found, creating a new one ...")
                    vectorstore = get_vectorstore(documents)
                    st.session_state.vectorstore = vectorstore
                else:
                    logger.debug("Vectorstore exists, adding new docs to it")
                    _ = st.session_state.vectorstore.add_documents(documents_chunks)
                chain = get_chain(st.session_state.vectorstore)
                st.session_state.chain = chain


def init_app():
    """All initialization stuffs"""
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":open_file_folder:")
    st.header("Chat with your PDF documents :open_file_folder:")

    if "messages" not in st.session_state:
        st.session_state.messages = []


def st_display_message_handler():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    """All streamlit related stuffs"""

    init_app()
    st_file_handler()
    st_display_message_handler()

    if prompt := st.chat_input("Ask question about your document(s) ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = CLIENT.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
