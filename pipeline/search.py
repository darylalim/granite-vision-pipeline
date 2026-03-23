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


def clear_collection(collection: chromadb.Collection) -> None:
    """Delete all documents from a ChromaDB collection."""
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)


def _extract_text(element: dict) -> str | None:
    """Extract indexable text from a build_output element.

    For pictures: returns description text.
    For tables: returns markdown.
    Prepends caption if non-empty. Returns None if no text content.
    """
    text: str | None = None
    if element["type"] == "picture":
        desc = element["content"].get("description")
        if desc:
            text = desc["text"]
    elif element["type"] == "table":
        md = element["content"].get("markdown", "")
        if md:
            text = md

    if text is None:
        return None

    caption = element.get("caption", "")
    if caption:
        text = f"{caption}\n{text}"

    return text


def _chunk_text(
    text: str,
    model: SentenceTransformer,
    token_limit: int = 8000,
    chunk_size: int = 7000,
    overlap: int = 200,
) -> list[str]:
    """Split text into chunks if it exceeds the token limit.

    Splits on '. ' (sentence boundaries), falling back to '\\n' for content
    like table markdown. Returns [text] if within token_limit.
    """
    tokenizer = model.tokenizer
    token_count = len(tokenizer.encode(text))
    if token_count <= token_limit:
        return [text]

    used_sep = ". "
    parts = text.split(used_sep)
    if len(parts) <= 1:
        used_sep = "\n"
        parts = text.split(used_sep)
    if len(parts) <= 1:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = len(tokenizer.encode(part + used_sep))

        if current_tokens + part_tokens > chunk_size and current:
            chunks.append(used_sep.join(current))
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                p_tokens = len(tokenizer.encode(p))
                if overlap_tokens + p_tokens > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += p_tokens
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(part)
        current_tokens += part_tokens

    if current:
        chunks.append(used_sep.join(current))

    return chunks if chunks else [text]


def index_elements(
    elements: list[dict],
    source: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """Index elements from build_output() into ChromaDB.

    Extracts text from each element, embeds it, and stores in the collection.
    Skips elements with no text content. Uses upsert for idempotent re-indexing.
    Returns the number of documents indexed.
    """
    docs: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for element in elements:
        text = _extract_text(element)
        if text is None:
            continue

        reference = element["reference"]
        base_id = f"{source}:{reference}"
        meta = {
            "source": source,
            "type": element["type"],
            "element_number": element["element_number"],
            "reference": reference,
        }

        chunks = _chunk_text(text, model)
        if len(chunks) == 1:
            docs.append(chunks[0])
            ids.append(base_id)
            metadatas.append(meta)
        else:
            for i, chunk in enumerate(chunks):
                docs.append(chunk)
                ids.append(f"{base_id}:chunk_{i}")
                metadatas.append(meta)

    if not docs:
        return 0

    raw = model.encode(docs)
    embeddings = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(docs)
