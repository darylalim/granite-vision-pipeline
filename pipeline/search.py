"""Document search and RAG using embeddings and ChromaDB."""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = ".chroma"
COLLECTION_NAME = "elements"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-english-r2"


def create_embedding_model() -> SentenceTransformer:
    """Load the Granite embedding model for document search.

    Returns a SentenceTransformer instance. Streamlit pages should cache
    this via st.cache_resource at the call site.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def get_collection() -> chromadb.Collection:
    """Get the persistent ChromaDB collection for document search.

    Uses a module-level constant CHROMA_PATH for storage location.
    Streamlit pages should cache this via st.cache_resource at the call site.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
