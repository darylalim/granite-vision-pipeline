# Focus App on Granite Vision Intended Use

## Context

The Granite Vision 3.3 2B model card describes its intended use as "visual document understanding tasks, such as analyzing tables and charts, performing OCR, and answering questions based on document content." The app currently has five features, but only two closely match this intended use: PDF Extraction (analyzing tables and charts) and Multipage QA (answering questions based on document content). The remaining three — Image Segmentation, Document Parsing, and Document Search — use different models or apply Granite Vision outside its stated purpose.

## Goal

Strip the app down to PDF Extraction and Multipage QA. Remove all code, tests, dependencies, and UI related to Image Segmentation, Document Parsing, and Document Search. Remove the landing page and make PDF Extraction the default page.

## Approach

Surgical deletion in a single pass. The removed features are cleanly isolated — each has its own pipeline module, page, and test file — so the removal is straightforward.

## Files to Delete

| File | Reason |
|---|---|
| `pipeline/segmentation.py` | Segmentation logic + SAM integration |
| `pipeline/doctags.py` | Document parsing to doctags |
| `pipeline/search.py` | Embedding, indexing, ChromaDB, RAG |
| `pages/segmentation.py` | Segmentation UI page |
| `pages/doctags.py` | Document parsing UI page |
| `pages/search.py` | Document search UI page |
| `streamlit_home.py` | Landing page (unnecessary with two pages) |
| `tests/test_segmentation.py` | Tests for deleted segmentation module |
| `tests/test_doctags.py` | Tests for deleted doctags module |
| `tests/test_search.py` | Tests for deleted search module |
| `examples/sample.jpg` | Only used by segmentation and doctags demos |

## Files to Modify

### `streamlit_app.py`

Remove landing page, segmentation, doctags, and search entries. Make PDF Extraction the default page with the home icon.

After:
```python
import streamlit as st

st.set_page_config(page_title="Granite Vision Pipeline")

pg = st.navigation(
    [
        st.Page("pages/extraction.py", title="PDF Extraction", icon=":material/home:"),
        st.Page("pages/qa.py", title="Multipage QA"),
    ]
)
pg.run()
```

### `pipeline/__init__.py`

Remove imports and `__all__` entries for deleted modules:
- From `pipeline.doctags`: `export_markdown`, `generate_doctags`, `get_pdf_page_count`, `parse_doctags`, `render_pdf_pages`
- From `pipeline.models`: `create_doctags_model`
- From `pipeline.search`: `clear_collection`, `create_embedding_model`, `generate_answer`, `get_collection`, `index_elements`, `query_index`
- From `pipeline.segmentation`: `create_sam_model`, `draw_mask`, `segment`

Remaining re-exports:
- From `pipeline.config`: `convert`, `create_converter`
- From `pipeline.models`: `create_granite_vision_model`, `generate_response`
- From `pipeline.output`: `build_output`, `get_description`, `get_table_content`
- From `pipeline.qa`: `generate_qa_response`, `resize_for_qa`
- From `pipeline.utils`: `timed`, `temp_upload`

### `pipeline/models.py`

Remove `create_doctags_model()` function. Keep `_load_vision_model()`, `create_granite_vision_model()`, and `generate_response()`.

Update `create_granite_vision_model()` docstring — remove "shared across segmentation" reference.

### `pages/extraction.py`

Remove all indexing logic:
- Drop imports: `create_embedding_model`, `get_collection`, `index_elements`
- Remove `embedding_model` and `collection` cache lines
- Remove the `try/except` block calling `index_elements` after extraction
- Remove `model_embedding` session state tracking
- Remove `index_count` logic and embedding-related sidebar status entry
- Sidebar status shows only Docling model status

### `tests/test_models.py`

Remove `test_create_doctags_model_uses_correct_repo` test.

### `pyproject.toml`

Remove from `[project.dependencies]`:
- `chromadb`
- `sentence-transformers`

Update `description` to reflect two-feature app.

### `CLAUDE.md`

Update all sections to reflect two-feature app:
- Project Overview: only PDF Extraction and Multipage QA
- Architecture: remove pipeline modules, pages, and test references for deleted features
- Dependencies: remove `chromadb` and `sentence-transformers`
- Key Details: remove segmentation, doctags, search, and SAM references

### `README.md`

Update to reflect two-feature app.

## Files Unchanged

- `pipeline/config.py` — converter factory, used by extraction
- `pipeline/output.py` — element building, used by extraction
- `pipeline/qa.py` — multipage QA logic
- `pipeline/utils.py` — `temp_upload()`, `timed()`
- `pages/qa.py` — QA page
- `ui_helpers.py` — shared UI functions
- `examples/sample.pdf` — used by both remaining features
- `tests/test_config.py`, `tests/test_output.py`, `tests/test_utils.py`, `tests/test_qa.py`, `tests/test_ui_helpers.py`

## Dependencies After Change

Runtime:
- `docling[vlm]` — PDF parsing, table extraction, picture description
- `pypdfium2` — PDF page rendering and page counting
- `streamlit` — web UI framework
- `torch` — tensor operations and model inference
- `transformers` — Granite Vision model loading

Dev (unchanged):
- `pytest`, `ruff`, `ty`

## Testing

After all changes:
- `uv run ruff check .` — lint passes
- `uv run ruff format .` — formatting passes
- `uv run pytest` — all remaining tests pass
- `uv run streamlit run streamlit_app.py` — app starts, PDF Extraction is default page, Multipage QA accessible from sidebar
