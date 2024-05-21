from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from PyPDF2 import PdfReader


@dataclass
class PdfInfo:
    """A naive dataclass for storing pdf information"""

    doc_name: str
    page_info: Dict[int, str]  # dict with page number as key and the text as value

    @property
    def text(self):
        return " ".join(self.page_info.values())

    def as_documents(self) -> List[Document]:
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

        return cls(doc_name=path.name, page_info=page_info)


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


def format_response(response: dict) -> str:
    answer = response.get("answer", "")
    if not answer:
        return ""
    contexts = response.get("context", {})

    for idx, context in enumerate(contexts):
        doc_name = context.metadata["doc_name"]
        page_number = context.metadata["page_num"]
        # Append the metadata to the answer string
        answer += f"\n\n - Source ({idx+1}) - Document: {doc_name}, Page number: {page_number}"
    return answer


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

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


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
            response = st.session_state.chain.invoke({"input": prompt})
            answer = format_response(response)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
