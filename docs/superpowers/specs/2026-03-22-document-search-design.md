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

Added to `[project.dependencies]` in `pyproject.toml`:

- `sentence-transformers` — loads `ibm-granite/granite-embedding-english-r2` (149M params, 768-dim, 8K context, Apache 2.0)
- `chromadb` — persistent local vector store

### File changes summary

- **New:** `pipeline/search.py`, `pages/search.py`, `tests/test_search.py`
- **Modified:** `pipeline/__init__.py` (add new exports), `streamlit_app.py` (add auto-indexing), `pyproject.toml` (add dependencies), `.gitignore` (add `.chroma/`), `CLAUDE.md` (see updates below)

### CLAUDE.md updates

- **Pipeline section**: Add `pipeline/search.py` entry listing all 6 public functions (`create_embedding_model`, `get_collection`, `index_elements`, `query_index`, `generate_answer`, `clear_collection`)
- **UI section**: Add `pages/search.py` entry describing the search page
- **Key Details section**: Add notes on ChromaDB storage path (`.chroma/`), hybrid chunking strategy, model reuse from QA, `min_similarity` threshold, idempotent re-indexing
- **Dependencies section**: Add `sentence-transformers` and `chromadb` to runtime dependencies
- **Tests section**: Add `tests/test_search.py` entry describing test coverage
- **`pipeline/__init__.py` re-exports list**: Add the 6 new functions

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

This module is pure Python with no Streamlit imports, consistent with all other pipeline modules. Streamlit pages wrap these factory functions with `st.cache_resource` at their call sites.

### Public API

```python
create_embedding_model() -> SentenceTransformer
```

Loads `ibm-granite/granite-embedding-english-r2`. Plain factory function — Streamlit pages cache it via `st.cache_resource` at the call site (e.g., `embedding_model = st.cache_resource(create_embedding_model)`). Both `streamlit_app.py` and `pages/search.py` cache this independently; since `st.cache_resource` is global across all pages in a Streamlit app, the model is loaded only once.

```python
get_collection() -> chromadb.Collection
```

Returns a persistent ChromaDB collection named `"elements"` stored at `.chroma/` in the project directory. Creates the directory if needed. No `path` parameter — uses a module-level constant `CHROMA_PATH = ".chroma"` to avoid caching inconsistencies. Streamlit pages cache via `st.cache_resource` at the call site.

```python
index_elements(
    elements: list[dict],
    source: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int
```

Takes the `elements` array from `build_output()` plus the source filename. For each element:

- Extracts text from element dict: `element["content"]["description"]["text"]` for pictures, `element["content"]["markdown"]` for tables. Prepends `element["caption"]` if non-empty.
- If text exceeds 8K tokens (counted via `model.tokenizer`), splits into chunks of ~7K tokens with ~200 token overlap at sentence boundaries (split on `. ` then merge to target size; falls back to splitting on `\n` for content without sentence boundaries, such as table markdown). Each chunk gets ID `f"{source}:{reference}:chunk_{i}"`.
- For non-chunked elements, document ID is `f"{source}:{reference}"`.
- Embeds with the model, stores in ChromaDB via `collection.upsert()` with metadata (`source`, `type`, `element_number`, `reference`)
- Skips elements with no text content (no description for pictures, no markdown for tables)
- Returns the number of documents indexed

```python
query_index(
    question: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
    min_similarity: float = 0.3,
) -> list[dict]
```

Embeds the question, queries ChromaDB for top-N similar documents. ChromaDB returns cosine distances; `similarity` is computed as `1 - distance`. Filters out results below `min_similarity` threshold (default 0.3 is intentionally permissive to avoid missing relevant results; can be tuned). Returns results as list of dicts with `text`, `metadata`, and `similarity` keys. Returns empty list if collection is empty.

```python
generate_answer(
    question: str,
    context: list[dict],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> str
```

Constructs a text-only prompt and sends it to Granite Vision 3.3 2B. The processor/model pair comes from `create_qa_model()` in `pipeline/qa.py`. The vision capability goes unused but the model handles text prompts correctly. Uses `max_new_tokens=1024`.

**Prompt construction:** Builds the prompt string from the `context` list (output of `query_index`). Each context dict has `text`, `metadata`, and `similarity` keys. The element type label comes from `metadata["type"]` (either `"picture"` or `"table"`). Elements are numbered sequentially (1-based) in the order they appear in the context list. If `metadata["type"]` is missing, defaults to `"element"`.

**Conversation structure:**

```python
prompt = f"Use the following context from a document to answer the question.\n..."
conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
inputs = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
)
```

**Prompt template:**

```
Use the following context from a document to answer the question.
If the context does not contain enough information, say so.

Context:
[Element 1 - picture]: A bar chart showing...
[Element 2 - table]: | Quarter | Revenue | ...

Question: How did revenue change in Q4?
```

```python
clear_collection(collection: chromadb.Collection) -> None
```

Deletes all documents from the collection. Used by the "Clear Index" button on the search page.

### Exports in `pipeline/__init__.py`

Add to the re-exports: `create_embedding_model`, `get_collection`, `index_elements`, `query_index`, `generate_answer`, `clear_collection`.

## UI page: `pages/search.py`

### Layout

1. **Header** — title "Document Search" with brief description
2. **Status indicator** — shows how many documents are currently indexed (via `collection.count()`)
3. **Question input** — `st.text_input` for the search query
4. **Search button** — triggers retrieval + RAG generation (disabled when question is empty)
5. **Answer display** — the generated answer in a prominent block
6. **Source elements** — expanders showing the retrieved context chunks with their metadata (type, source filename, element number)
7. **Clear Index button** — resets the ChromaDB collection via `clear_collection()`, with confirmation to prevent accidental deletion

### Model caching

All factory functions are wrapped with `st.cache_resource` at the page level, consistent with every other page:

```python
embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
qa_model = st.cache_resource(create_qa_model)
```

### Integration with extraction page (`streamlit_app.py`)

Two changes to the existing extraction page:

1. **Store `build_output()` result** — currently `build_output()` is called inline inside `json.dumps()` for the download button. Refactor to store the result first:

```python
output = build_output(doc, duration_s)
```

Then use `output` for both the download button and indexing.

2. **Add `st.cache_resource` wrappers at module level** alongside the existing `converter = st.cache_resource(create_converter)`:

```python
embedding_model = st.cache_resource(create_embedding_model)
collection = st.cache_resource(get_collection)
```

3. **Add auto-indexing** after `st.success("Done.")`:

```python
try:
    count = index_elements(output["elements"], uploaded_file.name, embedding_model(), collection())
    if count > 0:
        st.info(f"Indexed {count} elements for search.")
    else:
        st.info("No indexable content found (no descriptions or tables).")
except Exception:
    st.warning("Indexing for search failed, but extraction completed successfully.")
```

Indexing failure does not block the extraction result.

## Error handling and edge cases

### Indexing

- Elements with no text content (no description for pictures, no markdown for tables) are skipped
- Duplicate indexing of the same file overwrites existing entries via `collection.upsert()` with idempotent document IDs
- Embedding model load failure does not block extraction — caught and shown as `st.warning`
- Concurrent indexing from multiple browser tabs: ChromaDB handles concurrent writes safely; no additional locking needed

### Search

- Empty collection: shows "No documents indexed yet. Extract a PDF first."
- No results above `min_similarity` threshold: answer states insufficient context rather than hallucinating
- Empty question: search button is disabled

### Storage

- ChromaDB persists to `.chroma/` in the project directory
- `.chroma/` is added to `.gitignore`
- "Clear Index" calls `clear_collection()` and requires confirmation

## Testing: `tests/test_search.py`

Follows existing patterns: test pure logic directly, mock models where needed. No model weights required. Embedding model is mocked with `@patch`. ChromaDB tests use an in-memory client.

### `index_elements` tests

- Indexes picture elements correctly (text from `content.description.text`)
- Indexes table elements correctly (text from `content.markdown`)
- Prepends caption to text when present
- Skips elements with no description/content
- Chunks text that exceeds 8K tokens, produces correct chunk IDs
- Returns correct count of indexed documents
- Stores correct metadata (source, type, element_number, reference)
- Re-indexing same source is idempotent (document count does not double)

### `query_index` tests

- Returns results sorted by relevance
- Respects `n_results` parameter
- Filters results below `min_similarity` threshold
- Returns empty list for empty collection
- Result dicts contain expected keys (text, metadata, similarity)

### `generate_answer` tests

- Constructs prompt via `apply_chat_template` with text-only content (no image entries)
- Includes element type labels in context
- Handles empty context list (generates response stating insufficient context)
- Uses `max_new_tokens=1024`
- Returns decoded string from model

### `clear_collection` tests

- Empties the collection
- Collection count is 0 after clearing

### `create_embedding_model` and `get_collection` tests

- `create_embedding_model` loads correct model path (mocked)
- `get_collection` creates persistent ChromaDB client
- `get_collection` returns collection named `"elements"`
