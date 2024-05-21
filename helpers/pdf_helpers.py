from dataclasses import dataclass
from typing import Dict, List

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
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
