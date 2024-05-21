from typing import List

from PyPDF2 import PdfReader


def get_pdf_text(pdf_docs: List[str]):
    """Loads text from pdf"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
