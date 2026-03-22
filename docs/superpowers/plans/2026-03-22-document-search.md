# Document Search & RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a document search and RAG feature that auto-indexes extracted PDF content into ChromaDB and answers natural language questions using retrieved context.

**Architecture:** New `pipeline/search.py` module (pure Python, 6 public functions) + new `pages/search.py` Streamlit page. Extraction page auto-indexes after processing. Embedding via `granite-embedding-english-r2`, vector storage via ChromaDB, answer generation via existing Granite Vision 3.3 2B.

**Tech Stack:** sentence-transformers, chromadb, transformers (AutoProcessor, AutoModelForVision2Seq)

**Spec:** `docs/superpowers/specs/2026-03-22-document-search-design.md`

---

### Task 1: Add dependencies and project config

**Files:**
- Modify: `pyproject.toml:6-12`
- Modify: `.gitignore:14`

- [ ] **Step 1: Add sentence-transformers and chromadb to pyproject.toml**

In `pyproject.toml`, add the two new dependencies to the `dependencies` list:

```python
dependencies = [
    "chromadb",
    "docling[vlm]",
    "pypdfium2",
    "sentence-transformers",
    "streamlit",
    "torch",
    "transformers",
]
```

- [ ] **Step 2: Add .chroma/ to .gitignore**

Append to `.gitignore`:

```
# Vector database
.chroma/
```

- [ ] **Step 3: Run uv sync to install new dependencies**

Run: `uv sync`
Expected: Dependencies resolve and install successfully.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "chore: add sentence-transformers and chromadb dependencies"
```

---

### Task 2: Implement `create_embedding_model` and `get_collection`

**Files:**
- Create: `pipeline/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write failing tests for create_embedding_model**

Create `tests/test_search.py`:

```python
"""Tests for the search module."""

from pathlib import Path
from unittest.mock import MagicMock, patch


# --- create_embedding_model tests ---


@patch("pipeline.search.SentenceTransformer")
def test_create_embedding_model_loads_correct_model(
    mock_st_cls: MagicMock,
) -> None:
    from pipeline.search import create_embedding_model

    model = create_embedding_model()

    mock_st_cls.assert_called_once_with("ibm-granite/granite-embedding-english-r2")
    assert model is mock_st_cls.return_value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py::test_create_embedding_model_loads_correct_model -v`
Expected: FAIL — `pipeline.search` does not exist yet.

- [ ] **Step 3: Write minimal implementation for create_embedding_model**

Create `pipeline/search.py`:

```python
"""Document search and RAG using embeddings and ChromaDB."""

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py::test_create_embedding_model_loads_correct_model -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for get_collection**

Append to `tests/test_search.py`:

```python
import chromadb


# --- get_collection tests ---


def test_get_collection_returns_collection() -> None:
    from pipeline.search import COLLECTION_NAME, get_collection

    collection = get_collection()
    assert collection.name == COLLECTION_NAME


def test_get_collection_uses_persistent_client(tmp_path: Path) -> None:
    from pipeline.search import get_collection

    with patch("pipeline.search.CHROMA_PATH", str(tmp_path / "test_chroma")):
        collection = get_collection()
        # Verify the directory was created
        assert (tmp_path / "test_chroma").exists()
        assert collection is not None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "get_collection" -v`
Expected: FAIL — `get_collection` not defined.

- [ ] **Step 7: Implement get_collection**

Add to `pipeline/search.py`:

```python
import chromadb


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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -k "get_collection" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "feat(search): add create_embedding_model and get_collection"
```

---

### Task 3: Implement `clear_collection`

**Files:**
- Modify: `pipeline/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write failing tests for clear_collection**

Append to `tests/test_search.py`:

```python
# --- clear_collection tests ---


def test_clear_collection_empties_collection() -> None:
    from pipeline.search import clear_collection

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_clear")
    collection.upsert(ids=["1", "2"], documents=["doc1", "doc2"])
    assert collection.count() == 2

    clear_collection(collection)
    assert collection.count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search.py::test_clear_collection_empties_collection -v`
Expected: FAIL — `clear_collection` not defined.

- [ ] **Step 3: Implement clear_collection**

Add to `pipeline/search.py`:

```python
def clear_collection(collection: chromadb.Collection) -> None:
    """Delete all documents from a ChromaDB collection."""
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search.py::test_clear_collection_empties_collection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "feat(search): add clear_collection"
```

---

### Task 4: Implement text extraction helper and `index_elements`

**Files:**
- Modify: `pipeline/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write failing tests for element text extraction**

Append to `tests/test_search.py`:

```python
# --- _extract_text helper tests ---


def test_extract_text_from_picture_description() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "",
        "content": {"description": {"text": "A bar chart.", "created_by": "model"}},
    }
    assert _extract_text(element) == "A bar chart."


def test_extract_text_from_table_markdown() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "table",
        "caption": "",
        "content": {
            "markdown": "| Col |\n|---|\n| val |",
            "data": {"columns": ["Col"], "rows": [["val"]]},
        },
    }
    assert _extract_text(element) == "| Col |\n|---|\n| val |"


def test_extract_text_prepends_caption() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "Figure 1: Revenue",
        "content": {"description": {"text": "A chart.", "created_by": "model"}},
    }
    assert _extract_text(element) == "Figure 1: Revenue\nA chart."


def test_extract_text_returns_none_for_no_description() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "",
        "content": {"description": None},
    }
    assert _extract_text(element) is None


def test_extract_text_returns_none_for_empty_markdown() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "table",
        "caption": "",
        "content": {"markdown": "", "data": {"columns": [], "rows": []}},
    }
    assert _extract_text(element) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "_extract_text" -v`
Expected: FAIL — `_extract_text` not defined.

- [ ] **Step 3: Implement _extract_text helper**

Add to `pipeline/search.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -k "_extract_text" -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for index_elements**

Append to `tests/test_search.py`:

```python
# --- index_elements tests ---


def test_index_elements_picture() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_index_pic")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "A chart.", "created_by": "model"}},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 1
    assert collection.count() == 1

    result = collection.get(ids=["test.pdf:#/pictures/0"])
    assert result["documents"][0] == "A chart."
    assert result["metadatas"][0]["source"] == "test.pdf"
    assert result["metadatas"][0]["type"] == "picture"
    assert result["metadatas"][0]["element_number"] == 1
    assert result["metadatas"][0]["reference"] == "#/pictures/0"


def test_index_elements_table() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_index_table")

    elements = [
        {
            "element_number": 1,
            "type": "table",
            "reference": "#/tables/0",
            "caption": "",
            "content": {
                "markdown": "| A |\n|---|\n| 1 |",
                "data": {"columns": ["A"], "rows": [["1"]]},
            },
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 1
    assert collection.get(ids=["test.pdf:#/tables/0"])["documents"][0] == "| A |\n|---|\n| 1 |"


def test_index_elements_skips_no_content() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_skip")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": None},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 0
    assert collection.count() == 0


def test_index_elements_correct_metadata() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_meta")

    elements = [
        {
            "element_number": 3,
            "type": "picture",
            "reference": "#/pictures/2",
            "caption": "Fig 3",
            "content": {"description": {"text": "Diagram.", "created_by": "m"}},
        }
    ]

    index_elements(elements, "report.pdf", mock_model, collection)
    meta = collection.get(ids=["report.pdf:#/pictures/2"])["metadatas"][0]
    assert meta == {
        "source": "report.pdf",
        "type": "picture",
        "element_number": 3,
        "reference": "#/pictures/2",
    }


def test_index_elements_idempotent() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_idempotent")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "A chart.", "created_by": "m"}},
        }
    ]

    index_elements(elements, "test.pdf", mock_model, collection)
    index_elements(elements, "test.pdf", mock_model, collection)
    assert collection.count() == 1


def test_index_elements_returns_count() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_count")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "Pic 1.", "created_by": "m"}},
        },
        {
            "element_number": 2,
            "type": "table",
            "reference": "#/tables/0",
            "caption": "",
            "content": {
                "markdown": "| A |\n|---|\n| 1 |",
                "data": {"columns": ["A"], "rows": [["1"]]},
            },
        },
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 2


def test_index_elements_chunks_long_text() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    # Mock tokenizer to simulate text exceeding 8K tokens
    mock_tokenizer = MagicMock()
    call_count = 0

    def fake_encode(text: str) -> list[int]:
        nonlocal call_count
        call_count += 1
        # First call checks total length (>8000 to trigger chunking)
        # Subsequent calls for individual parts return small counts
        if "Long sentence" in text and text.count(". ") > 5:
            return list(range(9000))
        return list(range(100))

    mock_tokenizer.encode = fake_encode
    mock_model.tokenizer = mock_tokenizer
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_chunk")

    # Create element with text that will be chunked
    long_text = ". ".join([f"Long sentence {i}" for i in range(100)])
    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": long_text, "created_by": "m"}},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count >= 2  # Should produce multiple chunks

    # Verify chunk IDs follow the expected pattern
    all_ids = collection.get()["ids"]
    chunk_ids = [id for id in all_ids if "chunk_" in id]
    assert len(chunk_ids) >= 2
    assert "test.pdf:#/pictures/0:chunk_0" in all_ids
    assert "test.pdf:#/pictures/0:chunk_1" in all_ids
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "index_elements" -v`
Expected: FAIL — `index_elements` not defined.

- [ ] **Step 7: Implement index_elements**

Add to `pipeline/search.py`:

```python
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

    # Try sentence splitting first, fall back to newline
    used_sep = ". "
    parts = text.split(used_sep)
    if len(parts) <= 1:
        used_sep = "\n"
        parts = text.split(used_sep)
    if len(parts) <= 1:
        # No separator found — return as single chunk
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = len(tokenizer.encode(part + used_sep))

        if current_tokens + part_tokens > chunk_size and current:
            chunks.append(used_sep.join(current))
            # Overlap: keep last portion
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

    embeddings = model.encode(docs).tolist()
    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(docs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -k "index_elements" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "feat(search): add index_elements with text extraction and chunking"
```

---

### Task 5: Implement `query_index`

**Files:**
- Modify: `pipeline/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write failing tests for query_index**

Append to `tests/test_search.py`:

```python
# --- query_index tests ---


def test_query_index_returns_results() -> None:
    from pipeline.search import query_index

    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        [[0.1] * 768],  # for indexing
        [[0.1] * 768],  # for query
    ]

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "test_query", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=["doc1"],
        documents=["Revenue grew 20% in Q4."],
        embeddings=[[0.1] * 768],
        metadatas=[{"source": "test.pdf", "type": "picture", "element_number": 1, "reference": "#/pictures/0"}],
    )

    results = query_index("What happened to revenue?", mock_model, collection)
    assert len(results) == 1
    assert results[0]["text"] == "Revenue grew 20% in Q4."
    assert results[0]["metadata"]["source"] == "test.pdf"
    assert "similarity" in results[0]


def test_query_index_respects_n_results() -> None:
    from pipeline.search import query_index

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.5] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "test_n_results", metadata={"hnsw:space": "cosine"}
    )
    for i in range(5):
        collection.upsert(
            ids=[f"doc{i}"],
            documents=[f"Document {i}"],
            embeddings=[[0.5 + i * 0.01] * 768],
            metadatas=[{"source": "t.pdf", "type": "picture", "element_number": i, "reference": f"#/pictures/{i}"}],
        )

    results = query_index("query", mock_model, collection, n_results=2)
    assert len(results) <= 2


def test_query_index_empty_collection() -> None:
    from pipeline.search import query_index

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_empty_query")

    results = query_index("anything", mock_model, collection)
    assert results == []


def test_query_index_result_keys() -> None:
    from pipeline.search import query_index

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "test_keys", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=["doc1"],
        documents=["Some text"],
        embeddings=[[0.1] * 768],
        metadatas=[{"source": "a.pdf", "type": "table", "element_number": 1, "reference": "#/tables/0"}],
    )

    results = query_index("query", mock_model, collection)
    assert len(results) == 1
    assert set(results[0].keys()) == {"text", "metadata", "similarity"}


def test_query_index_filters_by_min_similarity() -> None:
    from pipeline.search import query_index

    mock_model = MagicMock()
    # Query embedding is orthogonal to one doc, similar to another
    mock_model.encode.return_value = [[1.0] + [0.0] * 767]

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        "test_sim_filter", metadata={"hnsw:space": "cosine"}
    )
    # Similar document (same direction)
    collection.upsert(
        ids=["similar"],
        documents=["Similar text"],
        embeddings=[[1.0] + [0.0] * 767],
        metadatas=[{"source": "a.pdf", "type": "picture", "element_number": 1, "reference": "#/pictures/0"}],
    )
    # Dissimilar document (orthogonal direction)
    collection.upsert(
        ids=["dissimilar"],
        documents=["Different text"],
        embeddings=[[0.0, 1.0] + [0.0] * 766],
        metadatas=[{"source": "a.pdf", "type": "table", "element_number": 2, "reference": "#/tables/0"}],
    )

    # With high threshold, only the similar doc should be returned
    results = query_index("query", mock_model, collection, n_results=5, min_similarity=0.9)
    assert len(results) == 1
    assert results[0]["text"] == "Similar text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "query_index" -v`
Expected: FAIL — `query_index` not defined.

- [ ] **Step 3: Implement query_index**

Add to `pipeline/search.py`:

```python
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

    query_embedding = model.encode([question]).tolist()

    # Don't request more results than exist
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
            output.append({
                "text": doc,
                "metadata": meta,
                "similarity": similarity,
            })

    return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -k "query_index" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "feat(search): add query_index with similarity filtering"
```

---

### Task 6: Implement `generate_answer`

**Files:**
- Modify: `pipeline/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write failing tests for generate_answer**

Append to `tests/test_search.py`:

```python
import torch


# --- generate_answer tests ---


def test_generate_answer_prompt_structure() -> None:
    from pipeline.search import generate_answer

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor.decode.return_value = "The answer."

    context = [
        {
            "text": "Revenue grew 20%.",
            "metadata": {"type": "picture", "source": "test.pdf", "element_number": 1, "reference": "#/pictures/0"},
            "similarity": 0.8,
        },
        {
            "text": "| Q | Rev |\n|---|---|\n| Q4 | 1.2M |",
            "metadata": {"type": "table", "source": "test.pdf", "element_number": 2, "reference": "#/tables/0"},
            "similarity": 0.7,
        },
    ]

    result = generate_answer("How did revenue change?", context, mock_processor, mock_model)

    # Verify conversation structure
    call_args = mock_processor.apply_chat_template.call_args
    conversation = call_args[0][0]
    content = conversation[0]["content"]

    # Should be text-only (no image entries)
    assert len(content) == 1
    assert content[0]["type"] == "text"

    # Verify prompt contains context with type labels
    prompt_text = content[0]["text"]
    assert "[Element 1 - picture]" in prompt_text
    assert "[Element 2 - table]" in prompt_text
    assert "Revenue grew 20%." in prompt_text
    assert "How did revenue change?" in prompt_text

    # Verify apply_chat_template kwargs
    call_kwargs = mock_processor.apply_chat_template.call_args[1]
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["return_dict"] is True
    assert call_kwargs["return_tensors"] == "pt"

    assert result == "The answer."


def test_generate_answer_uses_max_new_tokens() -> None:
    from pipeline.search import generate_answer

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2]])
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_processor.decode.return_value = "answer"

    generate_answer("q", [{"text": "t", "metadata": {"type": "picture"}, "similarity": 0.8}], mock_processor, mock_model)

    _, gen_kwargs = mock_model.generate.call_args
    assert gen_kwargs["max_new_tokens"] == 1024


def test_generate_answer_empty_context() -> None:
    from pipeline.search import generate_answer

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2]])
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_processor.decode.return_value = "No context available."

    result = generate_answer("question", [], mock_processor, mock_model)

    # Should still call the model (prompt says "say so" if insufficient context)
    mock_model.generate.assert_called_once()
    assert result == "No context available."


def test_generate_answer_missing_type_defaults_to_element() -> None:
    from pipeline.search import generate_answer

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2]])
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    mock_processor.decode.return_value = "answer"

    context = [{"text": "some text", "metadata": {}, "similarity": 0.5}]
    generate_answer("q", context, mock_processor, mock_model)

    prompt_text = mock_processor.apply_chat_template.call_args[0][0][0]["content"][0]["text"]
    assert "[Element 1 - element]" in prompt_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -k "generate_answer" -v`
Expected: FAIL — `generate_answer` not defined.

- [ ] **Step 3: Implement generate_answer**

Add to `pipeline/search.py`:

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


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
    # Build context section
    context_lines: list[str] = []
    for i, item in enumerate(context, 1):
        type_label = item["metadata"].get("type", "element")
        context_lines.append(f"[Element {i} - {type_label}]: {item['text']}")

    context_str = "\n".join(context_lines) if context_lines else "(No context available)"

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -k "generate_answer" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/search.py tests/test_search.py
git commit -m "feat(search): add generate_answer with RAG prompt construction"
```

---

### Task 7: Update `pipeline/__init__.py` exports

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Add search module exports**

Update `pipeline/__init__.py` to add the search module imports and __all__ entries:

Add import block after the existing imports:

```python
from pipeline.search import (
    clear_collection,
    create_embedding_model,
    generate_answer,
    get_collection,
    index_elements,
    query_index,
)
```

Add to `__all__` list (maintaining alphabetical order):

```python
"clear_collection",
"create_embedding_model",
"generate_answer",
"get_collection",
"index_elements",
"query_index",
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from pipeline import create_embedding_model, get_collection, index_elements, query_index, generate_answer, clear_collection; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run all existing tests to ensure no regressions**

Run: `uv run pytest -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pipeline/__init__.py
git commit -m "feat(search): export search functions from pipeline package"
```

---

### Task 8: Integrate auto-indexing into extraction page

**Files:**
- Modify: `streamlit_app.py:1-88`

- [ ] **Step 1: Refactor build_output call to store result**

In `streamlit_app.py`, change the import line (line 9) to also import search functions:

```python
from pipeline import (
    build_output,
    convert,
    create_converter,
    create_embedding_model,
    get_collection,
    get_description,
    index_elements,
)
```

Add module-level cache wrappers after line 11 (`converter = st.cache_resource(create_converter)`):

```python
embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
```

Replace the `st.download_button` block (lines 43-48) with:

```python
        output = build_output(doc, duration_s)

        st.download_button(
            label="Download JSON",
            data=json.dumps(output, indent=2),
            file_name=f"{uploaded_file.name}_annotations.json",
            mime="application/json",
        )

        try:
            count = index_elements(
                output["elements"], uploaded_file.name, embedding_model(), collection()
            )
            if count > 0:
                st.info(f"Indexed {count} elements for search.")
            else:
                st.info("No indexable content found (no descriptions or tables).")
        except Exception:
            st.warning(
                "Indexing for search failed, but extraction completed successfully."
            )
```

- [ ] **Step 2: Run linter to verify formatting**

Run: `uv run ruff check streamlit_app.py && uv run ruff format --check streamlit_app.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat(search): auto-index extracted elements on PDF extraction page"
```

---

### Task 9: Create search UI page

**Files:**
- Create: `pages/search.py`

- [ ] **Step 1: Create the search page**

Create `pages/search.py`:

```python
import streamlit as st

from pipeline import (
    clear_collection,
    create_embedding_model,
    create_qa_model,
    generate_answer,
    get_collection,
    query_index,
)

embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
qa_model = st.cache_resource(create_qa_model)

st.title("Document Search")
st.write(
    "Search across extracted document content using natural language questions. "
    "Documents are automatically indexed when processed on the PDF Extraction page."
)

coll = collection()
doc_count = coll.count()
st.metric("Indexed documents", doc_count)

if doc_count == 0:
    st.info("No documents indexed yet. Extract a PDF first.")

question = st.text_input("Question", placeholder="e.g., What does the revenue chart show?")

if st.button("Search", type="primary", disabled=not question or doc_count == 0):
    model = embedding_model()

    with st.spinner("Searching..."):
        results = query_index(question, model, coll)

    if not results:
        st.warning("No relevant results found for your question.")
    else:
        processor, gen_model = qa_model()

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, results, processor, gen_model)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            type_label = meta.get("type", "element").capitalize()
            source = meta.get("source", "Unknown")
            elem_num = meta.get("element_number", "?")
            similarity = result["similarity"]

            with st.expander(
                f"{type_label} (Element {elem_num}) from {source} — similarity: {similarity:.2f}",
                expanded=i == 1,
            ):
                st.text(result["text"])

st.divider()

with st.popover("Clear Index", disabled=doc_count == 0):
    st.write(f"This will delete all {doc_count} indexed documents. Are you sure?")
    if st.button("Confirm Clear", type="primary"):
        clear_collection(coll)
        st.success("Index cleared.")
        st.rerun()
```

- [ ] **Step 2: Run linter to verify formatting**

Run: `uv run ruff check pages/search.py && uv run ruff format --check pages/search.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add pages/search.py
git commit -m "feat(search): add document search UI page with RAG answers"
```

---

### Task 10: Run full test suite and lint

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (existing + new).

- [ ] **Step 2: Run linter on entire project**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No lint or format issues.

- [ ] **Step 3: Run type checker**

Run: `uv run ty check .`
Expected: No new type errors introduced (pre-existing errors are acceptable).

- [ ] **Step 4: Fix any issues found in steps 1-3**

If any tests, lint, or type errors were introduced, fix them and re-run.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix(search): address lint and test issues"
```

(Skip this step if no fixes were needed.)

---

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Pipeline section**

Add entry for `pipeline/search.py`:

```markdown
- `pipeline/search.py` — `create_embedding_model()` factory, `get_collection()` ChromaDB factory, `index_elements()` embedding and storage, `query_index()` similarity search, `generate_answer()` RAG generation, `clear_collection()` index reset
```

- [ ] **Step 2: Update UI section**

Add entry for `pages/search.py`:

```markdown
- `pages/search.py` — document search page; question input, RAG answer display with source elements, index status and clear button
```

- [ ] **Step 3: Update Key Details section**

Add search-specific details:

```markdown
- Search uses `ibm-granite/granite-embedding-english-r2` (sentence-transformers) for embeddings, stored in ChromaDB at `.chroma/`
- Elements are indexed per-element from `build_output()` result; elements exceeding 8K tokens are chunked with ~200 token overlap
- `generate_answer()` reuses `create_qa_model()` from QA module with a text-only prompt (no images)
- Re-indexing the same PDF is idempotent via `collection.upsert()` with `source:reference` document IDs
- `query_index()` filters by cosine similarity threshold (default 0.3)
```

- [ ] **Step 4: Update Dependencies section**

Add to runtime dependencies:

```markdown
- `sentence-transformers` — embedding model loading for document search
- `chromadb` — persistent local vector database for search index
```

- [ ] **Step 5: Update Tests section**

Add entry:

```markdown
- `tests/test_search.py` — `create_embedding_model()` and `get_collection()` with mocks, `index_elements()` with various element types and idempotency, `query_index()` with similarity filtering, `generate_answer()` prompt structure and model interaction, `clear_collection()` verification; uses in-memory ChromaDB client, no model weights required
```

- [ ] **Step 6: Update pipeline/__init__.py re-exports list**

Update the re-exports list in the Architecture section to include: `clear_collection`, `create_embedding_model`, `generate_answer`, `get_collection`, `index_elements`, `query_index`.

- [ ] **Step 7: Update project description**

Update the description at the top to mention the search feature:

```markdown
5. **Document Search** — search across extracted content and get RAG-powered answers using [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) + [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)
```

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for document search feature"
```

---

### Task 12: Version bump

**Files:**
- Modify: `pyproject.toml:3`

- [ ] **Step 1: Bump version to 0.5.0**

Update version in `pyproject.toml`:

```toml
version = "0.5.0"
```

- [ ] **Step 2: Update project description**

Update description in `pyproject.toml`:

```toml
description = "Extract and describe pictures and tables in PDFs, segment objects in images, generate doctags from document images, answer questions across multiple document pages, and search extracted content with RAG"
```

- [ ] **Step 3: Update lockfile**

Run: `uv sync`
Expected: Lockfile updated for new version.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump version to 0.5.0 for document search feature"
```
