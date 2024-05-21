import streamlit as st
from langchain.chains import conversational_retrieval
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from PyPDF2 import PdfReader

EMBEDDINGS = OpenAIEmbeddings(show_progress_bar=True)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        logger.info(f"Loading {pdf}")
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            _text = page.extract_text()
            text += _text
            logger.info(f"Loaded total {len(_text)} chars")
    return text


def get_text_chunks(text):
    logger.info("Splitting into chunks ...")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    logger.info(f"Splitted into {len(chunks)} chunks.")
    return chunks


def get_vectorstore(text_chunks):
    logger.info("Creating Embeddings")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=EMBEDDINGS)
    logger.info("Sucessfully created a vectorstore")
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = conversational_retrieval(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    logger.debug(f"output: {conversation_chain}")
    return conversation_chain


def user_input(user_question: str, chain):
    docs = st.session_state.vectorstore.similarity_search(user_question)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    logger.debug(f"output {response['output_text']}")
    st.write("Reply: ", response["output_text"])


def main():
    """All streamlit related stuffs"""
    st.header("Chat with your PDF documents :open_file_folder:")
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":open_file_folder:")

    # initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

                if "vectorstore" not in st.session_state:
                    st.session_state.vectorstore = None
                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                chain = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
