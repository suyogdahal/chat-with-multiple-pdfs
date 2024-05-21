"""All streamlit related handlers here"""

import streamlit as st


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
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error(
                        "Unable to extract text. Please provide a digital PDF. Our service does not currently support OCR for non-digital PDF documents."
                    )
                    return
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                chain = get_chain(vectorstore)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
