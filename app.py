import streamlit as st
from loguru import logger

from helpers.langchain_helpers import get_chain, get_vectorstore
from helpers.misc import format_response
from helpers.pdf_helpers import get_documents, get_pdf_infos, get_splitted_docs


def init_app():
    """All initialization stuffs"""
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":open_file_folder:")
    st.header("Chat with your PDF documents :open_file_folder:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


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
