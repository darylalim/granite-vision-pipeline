"""Document search and RAG using embeddings and ChromaDB."""

from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForVision2Seq, AutoProcessor

CHROMA_PATH = str(Path(__file__).resolve().parent.parent / ".chroma")
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
    ids = collection.get(include=[])["ids"]
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


def query_index(
    question: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
    min_similarity: float = 0.3,
) -> list[dict]:
    """Search the index for documents relevant to a question.

    Embeds the question and queries ChromaDB. Filters results below
    min_similarity (cosine similarity = 1 - distance). Returns list
    of dicts with 'text', 'metadata', and 'similarity' keys.
    """
    if collection.count() == 0:
        return []

    raw = model.encode([question])
    query_embedding = raw.tolist() if hasattr(raw, "tolist") else list(raw)

    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - dist
        if similarity >= min_similarity:
            output.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity,
                }
            )

    return output


def generate_answer(
    question: str,
    context: list[dict],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str:
    """Generate a RAG answer using retrieved context and Granite Vision.

    Constructs a text-only prompt from the context and question, sends to
    the model via apply_chat_template. Uses max_new_tokens=1024.
    """
    context_lines: list[str] = []
    for i, item in enumerate(context, 1):
        type_label = item["metadata"].get("type", "element")
        context_lines.append(f"[Element {i} - {type_label}]: {item['text']}")

    context_str = (
        "\n".join(context_lines) if context_lines else "(No context available)"
    )

    prompt = (
        "Use the following context from a document to answer the question.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(  # type: ignore[operator]
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1024)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(trimmed[0], skip_special_tokens=True)  # type: ignore[operator]
    return decoded
