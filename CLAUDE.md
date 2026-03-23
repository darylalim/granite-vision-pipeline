# Granite Vision Pipeline

## Project Overview

Streamlit web app with five capabilities:

1. **PDF Extraction** — extract and describe pictures and tables in PDF documents using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)
2. **Image Segmentation** — segment objects in images using natural language prompts, with [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) + [SAM](https://huggingface.co/facebook/sam-vit-huge) refinement
3. **DocTags Generation** — parse document images and PDFs to structured text in doctags format using [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)
4. **Multipage QA** — answer questions across up to 8 document pages using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) with images resized to 768px max dimension
5. **Document Search** — search across extracted content and get RAG-powered answers using [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) + [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

```bash
uv run ruff check .                          # lint
uv run ruff format .                         # format
uv run ty check .                            # type check
uv run pytest                                # run all tests
uv run pytest tests/test_file.py::test_name  # run single test
```

## Code Style

- `snake_case` for functions/variables, `PascalCase` for classes
- Type annotations on all parameters and returns

## Dependencies

Runtime (`[project.dependencies]`):
- `chromadb` — persistent local vector database for search index
- `docling[vlm]` — PDF parsing, table extraction, VLM-based picture description
- `pypdfium2` — PDF page rendering and page counting
- `sentence-transformers` — embedding model loading for document search
- `streamlit` — web UI framework
- `torch` — tensor operations and model inference
- `transformers` — model loading (Granite Vision, SAM, Granite Docling)

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

Overrides (`[tool.uv]`):
- `opencv-python` replaced with `opencv-python-headless` to avoid duplicate `libavdevice` dylib conflicts with `av` on macOS

## Architecture

### Pipeline

- `pipeline/__init__.py` — re-exports public API
- `pipeline/models.py` — `_load_vision_model()` shared loader; `create_granite_vision_model()` for Granite Vision 3.3 2B (shared across segmentation, QA, and RAG); `create_doctags_model()` for Granite Docling 258M; `generate_response()` shared generate-trim-decode helper
- `pipeline/utils.py` — `temp_upload()` context manager for temporary file handling; `timed()` context manager for wall-clock timing
- `pipeline/config.py` — `create_converter()` factory; `convert()` wrapper; warning filters for upstream docling/transformers deprecations
- `pipeline/output.py` — `build_output()` produces a unified `elements` array from pictures and tables; `get_description()` extracts picture descriptions from `meta` with fallback to `annotations`; `get_table_content()` extracts table markdown and structured column/row data
- `pipeline/segmentation.py` — `segment()` runs Granite Vision referring segmentation + SAM refinement via `generate_response()`; `draw_mask()` for overlay visualization; `create_sam_model()` factory; internal helpers for RLE parsing, mask processing, point sampling, and logit computation
- `pipeline/doctags.py` — `generate_doctags()` inference (different generation pattern, not using `generate_response()`); `parse_doctags()` conversion to DoclingDocument; `export_markdown()` wrapper; `render_pdf_pages()` PDF-to-image rendering; `get_pdf_page_count()` for page count without rendering
- `pipeline/qa.py` — `resize_for_qa()` image resizing; `generate_qa_response()` multi-image QA (delegates to `generate_response()`)
- `pipeline/search.py` — `create_embedding_model()` factory; `get_collection()` ChromaDB factory; `index_elements()` embedding and storage; `query_index()` similarity search; `generate_answer()` RAG generation (delegates to `generate_response()`); `clear_collection()` index reset

### UI

- `streamlit_app.py` — PDF extraction page; file upload, annotation, per-picture and per-table preview in expanders
- `pages/segmentation.py` — segmentation page; image upload, text prompt, mask overlay preview, mask download
- `pages/doctags.py` — doctags generation page; image/PDF upload, raw doctags display, markdown preview, per-page expanders for PDFs
- `pages/qa.py` — multipage QA page; PDF/image upload, page selection, question input, answer display with thumbnails
- `pages/search.py` — document search page; question input, RAG answer display with source elements, index status and clear button

### Key Details

- `convert()` accepts an optional `converter` parameter to reuse a cached instance, avoiding model reload on each call
- `get_description()` falls back to `pic.annotations` because docling appends `DescriptionAnnotation` after `PictureItem` construction, so the `meta` migration validator doesn't run
- Output JSON contains `document_info` (picture count, table count, timing) and an `elements` array with `type` discriminator (`"picture"` or `"table"`) and type-specific `content`
- Segmentation, QA, and RAG answer generation share `create_granite_vision_model()`; when cached via `st.cache_resource`, one model instance is loaded across pages
- Segmentation loads Granite Vision and SAM model instances (not shared with docling's internal model)
- DocTags generation uses `ibm-granite/granite-docling-258M` loaded directly via Transformers (not Docling's VlmPipeline), with prompt `"Convert this page to docling."`
- For PDFs in doctags, pages are rendered to images via pypdfium2 at 144 DPI, then each page is processed independently
- Adding `pages/` directory activates Streamlit multipage navigation with sidebar
- All models are cached via `st.cache_resource` at the page level
- QA images are resized so the longer dimension is 768px to stay within GPU memory limits for up to 8 pages
- PDF page count is obtained via `get_pdf_page_count()` without rendering; only selected pages are rendered via `render_pdf_pages(page_indices=...)`
- QA page validates uploads: rejects mixed PDF + image uploads and multiple PDFs
- Search uses `ibm-granite/granite-embedding-english-r2` (sentence-transformers) for embeddings, stored in ChromaDB at `.chroma/`
- Elements are indexed per-element from `build_output()` result; elements exceeding 8K tokens are chunked with ~200 token overlap
- `generate_answer()` uses `generate_response()` with a text-only prompt (no images)
- Re-indexing the same PDF is idempotent via `collection.upsert()` with `source:reference` document IDs
- `query_index()` filters by cosine similarity threshold (default 0.3)

## Tests

- `tests/test_config.py` — `create_converter()` factory with pipeline option verification; `convert()` with and without provided converter
- `tests/test_output.py` — `build_output()`, `build_element()`, `get_description()`, `get_table_content()` with real Docling objects; covers pictures, tables, mixed documents, annotations fallback, meta priority
- `tests/test_models.py` — `_load_vision_model()`, `create_granite_vision_model()`, `create_doctags_model()`, `generate_response()` with mocked transformers classes; no model weights required
- `tests/test_utils.py` — `temp_upload()` file creation, cleanup, and exception safety; `timed()` duration measurement
- `tests/test_segmentation.py` — `create_sam_model()`, `segment()`, `extract_segmentation()`, `prepare_mask()`, `sample_points()`, `compute_logits_from_mask()`, `draw_mask()` unit tests; no model weights required
- `tests/test_doctags.py` — `render_pdf_pages()` with real PDF fixture; `get_pdf_page_count()`; `parse_doctags()` with sample doctags strings; `generate_doctags()` with mocked model; `export_markdown()` verification; no model weights required
- `tests/test_qa.py` — `resize_for_qa()` dimension and aspect ratio tests; `generate_qa_response()` prompt structure, validation, and delegation to `generate_response()`; no model weights required
- `tests/test_search.py` — `create_embedding_model()` and `get_collection()` with mocks; `index_elements()` with various element types and idempotency; `query_index()` with similarity filtering; `generate_answer()` prompt structure and delegation to `generate_response()`; `clear_collection()` verification; uses in-memory ChromaDB client, no model weights required

All tests import directly from `pipeline` — no Streamlit mocking needed.
