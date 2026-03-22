# Document Search & RAG Feature Design

## Overview

Add a document search and RAG (retrieval-augmented generation) feature to the Granite Vision Pipeline. PDFs processed on the extraction page are automatically indexed into a local vector store. A new search page lets users ask natural language questions and receive grounded answers based on retrieved content.

## Requirements

- **Single and multi-document search** — search across one or many previously extracted PDFs
- **Fully local** — no external APIs; embedding model and vector store run locally
- **RAG-style answers** — retrieve relevant content, then generate a natural language answer using Granite Vision 3.3 2B
- **Automatic indexing** — PDFs are indexed after extraction without user intervention
- **Hybrid chunking** — index per-element by default, chunk elements that exceed the embedding model's 8K token limit

## Architecture

Two new files following the existing one-module-one-page pattern:

```
pipeline/search.py    — embedding, indexing, retrieval, RAG generation
pages/search.py       — search UI page
```

The PDF extraction page (`streamlit_app.py`) gains automatic indexing after successful extraction.

### New dependencies

- `sentence-transformers` — loads `ibm-granite/granite-embedding-english-r2` (149M params, 768-dim, 8K context, Apache 2.0)
- `chromadb` — persistent local vector store

### Data flow

```
PDF Extraction page                    Search page
─────────────────                      ───────────
Upload PDF                             Enter question
    ↓                                      ↓
convert() → DoclingDocument            query_index() → relevant chunks
    ↓                                      ↓
build_output() → elements              generate_answer(question, chunks, model)
    ↓                                      ↓
index_elements(elements, source)       Display answer + source elements
    ↓
ChromaDB (persistent, .chroma/)
```

## Pipeline module: `pipeline/search.py`

### Public API

```python
create_embedding_model() -> SentenceTransformer
```

Loads `ibm-granite/granite-embedding-english-r2`. Cached via `st.cache_resource` at the app level (shared between extraction page indexing and search page querying).

```python
get_collection(path: str = ".chroma") -> chromadb.Collection
```

Returns a persistent ChromaDB collection named `"elements"`. Creates the DB directory if it doesn't exist. Cached via `st.cache_resource` at the app level.

```python
index_elements(
    elements: list[dict],
    source: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int
```

Takes the `elements` array from `build_output()` plus the source filename. For each element:

- Builds a text representation: description text for pictures, markdown for tables, with caption prepended if present
- If text exceeds 8K tokens, splits into chunks at sentence boundaries
- Embeds with the model, stores in ChromaDB with metadata (`source`, `type`, `element_number`, `reference`)
- Uses `source + reference` as ChromaDB document ID for idempotent re-indexing
- Skips elements with no text content
- Returns the number of documents indexed

```python
query_index(
    question: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
) -> list[dict]
```

Embeds the question, queries ChromaDB for top-N similar documents. Returns results with text content and metadata.

```python
generate_answer(
    question: str,
    context: list[dict],
    processor: Any,
    model: Any,
) -> str
```

Constructs a text-only prompt with retrieved context and question, sends to Granite Vision 3.3 2B (reusing `create_qa_model()` from `pipeline/qa.py`). Returns the generated answer.

Prompt template:

```
Use the following context from a document to answer the question.

Context:
[Element 1 - picture]: A bar chart showing...
[Element 2 - table]: | Quarter | Revenue | ...

Question: How did revenue change in Q4?
```

## UI page: `pages/search.py`

### Layout

1. **Header** — title "Document Search" with brief description
2. **Status indicator** — shows how many documents are currently indexed (via `collection.count()`)
3. **Question input** — `st.text_input` for the search query
4. **Search button** — triggers retrieval + RAG generation (disabled when question is empty)
5. **Answer display** — the generated answer in a prominent block
6. **Source elements** — expanders showing the retrieved context chunks with their metadata (type, source filename, element number)
7. **Clear Index button** — resets the ChromaDB collection, with confirmation to prevent accidental deletion

### Model caching

- **Embedding model**: cached at app level, shared between extraction page (indexing) and search page (querying)
- **ChromaDB collection**: cached at app level, same collection accessed from both pages
- **Granite Vision for RAG**: reuses `create_qa_model()` from `pipeline/qa.py`, cached on the search page

### Integration with extraction page

After the existing `st.success("Done.")` in `streamlit_app.py`, add automatic indexing:

```python
index_elements(elements, uploaded_file.name, embedding_model(), collection())
st.info(f"Indexed {len(elements)} elements for search.")
```

Indexing failure shows `st.warning` but does not block the extraction result.

## Error handling and edge cases

### Indexing

- Elements with no text content (no description, no markdown) are skipped
- Duplicate indexing of the same file overwrites existing entries via idempotent document IDs (`source + reference`)
- Embedding model load failure does not block extraction — shows `st.warning`

### Search

- Empty collection: shows "No documents indexed yet. Extract a PDF first."
- No relevant results above similarity threshold: answer states this rather than hallucinating
- Empty question: search button is disabled

### Storage

- ChromaDB persists to `.chroma/` in the project directory
- `.chroma/` is added to `.gitignore`
- "Clear Index" requires confirmation

## Testing: `tests/test_search.py`

Follows existing patterns: test pure logic directly, mock models where needed. No model weights required.

### `index_elements` tests

- Indexes picture elements correctly (text from description)
- Indexes table elements correctly (text from markdown)
- Prepends caption to text when present
- Skips elements with no description/content
- Chunks text that exceeds 8K tokens
- Returns correct count of indexed documents
- Stores correct metadata (source, type, element_number, reference)

### `query_index` tests

- Returns results sorted by relevance
- Respects `n_results` parameter
- Returns empty list for empty collection
- Result dicts contain expected keys (text, metadata)

### `generate_answer` tests

- Constructs prompt with context and question
- Includes element type labels in context
- Handles empty context list
- Returns decoded string from model

### `create_embedding_model` and `get_collection` tests

- `create_embedding_model` loads correct model path (mocked)
- `get_collection` creates persistent ChromaDB client at specified path
- `get_collection` returns same collection on repeated calls
