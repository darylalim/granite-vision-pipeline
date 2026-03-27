# Granite Vision Pipeline

## Project Overview

Streamlit web app for multipage document question answering using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Upload a PDF, select 2-8 consecutive pages via a thumbnail grid and range slider, ask a question, and get a text answer. Supports conversation history, source page verification, and Q&A export.

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
- `pypdfium2` -- PDF page rendering and page counting
- `streamlit` (>=1.55.0) -- web UI framework
- `torch` -- tensor operations and model inference
- `torchvision` -- fast image preprocessing for vision models
- `transformers` -- model loading (Granite Vision)

Dev (`[dependency-groups] dev`):
- `pytest` -- testing
- `ruff` -- linting and formatting
- `ty` -- type checking

## Architecture

### Pipeline

- `pipeline/__init__.py` -- re-exports public API
- `pipeline/models.py` -- `_load_vision_model()` shared loader; `create_granite_vision_model()` for Granite Vision 3.3 2B; `generate_response()` generate-trim-decode helper
- `pipeline/utils.py` -- `temp_upload()` context manager for temporary file handling; `timed()` context manager for wall-clock timing
- `pipeline/pdf.py` -- `render_pdf_pages()` PDF-to-image rendering; `get_pdf_page_count()` for page count without rendering
- `pipeline/qa.py` -- `resize_for_qa()` image resizing; `generate_qa_response()` multi-page QA requiring 2-8 consecutive pages (delegates to `generate_response()`)

### UI

- `streamlit_app.py` -- single-page app; PDF upload, thumbnail grid page selection with range slider (2-8 consecutive pages), text area question input, tabbed answer/source display with conversation history, Q&A export download
- `ui_helpers.py` -- shared UI functions: `show_upload_preview()` for PDF file info, `clamp_page_range()` for page range validation, `render_thumbnail_grid()` for page thumbnail display, `format_qa_export()` for markdown Q&A export, `load_example()` for demo mode files; no pipeline imports
- `examples/` -- sample files for demo mode

### Key Details

- Only PDF uploads are supported; the file uploader accepts a single PDF
- QA requires 2-8 consecutive pages; PDFs with fewer than 2 pages are rejected
- Page selection uses a range slider (explicit `value` tuple for reliable range mode) with thumbnail grid; all pages rendered at 72 DPI as thumbnails (batched for PDFs over 50 pages), cached in `st.session_state`
- Page images are resized so the longer dimension is 768px to stay within GPU memory limits
- PDF page count is obtained via `get_pdf_page_count()` without rendering; only selected pages are rendered at full DPI via `render_pdf_pages(page_indices=...)`
- Answer button is always clickable; validates on click (shows warnings for missing PDF/pages/question) instead of being disabled
- Answer displayed in tabbed view: Answer tab (with conversation history) and Source Pages tab (pre-resize page images)
- Conversation history stored in `st.session_state`; resets on new upload or page selection change
- Q&A sessions can be exported as markdown via download button
- Model cached via `st.cache_resource`; model status shown inline via spinner messages
- "Try with example" button uses `st.session_state` flags and `load_example()` from `ui_helpers.py`; user uploads clear the example flag and reset history

## Tests

- `tests/test_models.py` -- `_load_vision_model()`, `create_granite_vision_model()`, `generate_response()` with mocked transformers classes; no model weights required
- `tests/test_utils.py` -- `temp_upload()` file creation, cleanup, and exception safety; `timed()` duration measurement
- `tests/test_pdf.py` -- `render_pdf_pages()` with real PDF fixture; `get_pdf_page_count()`; no model weights required
- `tests/test_qa.py` -- `resize_for_qa()` dimension and aspect ratio tests; `generate_qa_response()` prompt structure, validation (2-8 images), and delegation to `generate_response()`; no model weights required
- `tests/test_ui_helpers.py` -- `_ExampleFile` BytesIO wrapper attributes; `load_example()` file loading, name, size, seekability, and real example file validation; `show_upload_preview()` PDF preview rendering; `clamp_page_range()` range validation and clamping; `render_thumbnail_grid()` column creation, image display, page captions, and selection highlighting; `format_qa_export()` markdown structure, Q&A pairs, and timestamp format

Pipeline tests import directly from `pipeline` -- no Streamlit mocking needed. UI helper tests mock `streamlit` via `unittest.mock.patch`.
