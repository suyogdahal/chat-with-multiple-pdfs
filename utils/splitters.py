from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
