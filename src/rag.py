"""
RAG: Build a ChromaDB vector store from the product catalog.

The vector store is created once at import time and re-used by the
search_snackstack_menu tool.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import embeddings, get_logger
from src.data import SNACKSTACK_MENU

logger = get_logger("rag")

def _build_documents() -> list[Document]:
    """Convert every catalog entry into a LangChain Document."""
    docs: list[Document] = []
    for p in SNACKSTACK_MENU:
        content = (
            f"Dish: {p['name']}\n"
            f"Cuisine: {p['cuisine']}\n"
            f"Category: {p['category']}\n"
            f"Price: ₹{p['price (INR)']}\n"
            f"Rating: {p['rating']}/5\n"
            f"Dietary Tags: {', '.join(p['dietary_tags'])}\n"
            f"Description: {p['description']}\n"
            f"Available: {'Yes' if p['availability'] else 'No'}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": p["id"],
                    "name": p["name"],
                    "cuisine": p["cuisine"],
                    "category": p["category"],
                    "price": p["price (INR)"],
                    "rating": p["rating"],
                },
            )
        )
    return docs


def build_vectorstore() -> Chroma:
    """Create an in-memory ChromaDB collection from the menu."""
    docs = _build_documents()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="snackstack_menu",
    )
    logger.info("Vector store ready  (%d products indexed)", len(docs))
    return store


# Module-level singleton so every importer shares the same store
menu_vectorstore = build_vectorstore()
