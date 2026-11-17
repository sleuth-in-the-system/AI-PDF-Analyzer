"""Core PDF processing and RAG helpers for the AI-Powered PDF Chatbot."""
from __future__ import annotations

from io import BytesIO
from typing import Iterable

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_MB = 25
MAX_FILES = 10


class PDFProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be processed safely."""


def validate_pdf_upload(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Only PDF files are accepted.")
    if not content:
        raise PDFProcessingError("The uploaded file is empty.")
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise PDFProcessingError(f"File exceeds the {MAX_FILE_MB} MB size limit.")
    if not content.startswith(b"%PDF"):
        raise PDFProcessingError("The file does not appear to be a valid PDF.")


def extract_pdf_pages(filename: str, content: bytes) -> list[Document]:
    """Extract non-empty page text and preserve 1-based page metadata."""
    validate_pdf_upload(filename, content)
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        pages: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_number},
                    )
                )
    except Exception as exc:
        raise PDFProcessingError(f"Could not read this PDF: {exc}") from exc

    if not pages:
        raise PDFProcessingError(
            "No selectable text was found. This PDF may be empty or scanned. "
            "OCR is not included in this version; please try a text-based PDF."
        )
    return pages


def split_pages(pages: Iterable[Document], chunk_size: int = 1000,
                chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(pages))
    return [c for c in chunks if c.page_content.strip()]


def format_sources(documents: list[Document]) -> list[str]:
    seen = set()
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "Unknown")
        label = f"{source} — page {page}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources



def build_rag_prompt(
    question: str,
    context: str,
    chat_history: str = ""
) -> str:
    return f"""You are a careful PDF study assistant.

Answer the user's question using ONLY the supplied document context.
Use the conversation history to understand references to previous
questions and answers, such as "it", "that concept", or "the second type".

The conversation history helps you understand the question.
It must NOT be treated as evidence about the document.

If the document context does not contain enough information,
say:
"Information not found in the uploaded document."

Do not invent facts, page numbers, quotations, or citations.
Use clear, concise language. Distinguish direct evidence from
reasonable interpretation.

CONVERSATION HISTORY:
{chat_history if chat_history else "No previous conversation."}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""
