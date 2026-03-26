# Granite Vision Pipeline

## Project Overview

Streamlit web app for multipage document question answering using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Upload a PDF or up to 8 images, ask a question, and get an answer across all pages. Images are resized to 768px max dimension.

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
- `pypdfium2` — PDF page rendering and page counting
- `streamlit` — web UI framework
- `torch` — tensor operations and model inference
- `torchvision` — fast image preprocessing for vision models
- `transformers` — model loading (Granite Vision)

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

## Architecture

### Pipeline

- `pipeline/__init__.py` — re-exports public API
- `pipeline/models.py` — `_load_vision_model()` shared loader; `create_granite_vision_model()` for Granite Vision 3.3 2B; `generate_response()` generate-trim-decode helper
- `pipeline/utils.py` — `temp_upload()` context manager for temporary file handling; `timed()` context manager for wall-clock timing
- `pipeline/pdf.py` — `render_pdf_pages()` PDF-to-image rendering; `get_pdf_page_count()` for page count without rendering
- `pipeline/qa.py` — `resize_for_qa()` image resizing; `generate_qa_response()` multi-image QA (delegates to `generate_response()`)

### UI

- `streamlit_app.py` — single-page app; PDF/image upload, "Try with example", page selection, question input, answer display with correct page numbering
- `ui_helpers.py` — shared UI functions: `show_upload_preview()` for file thumbnails, `show_help()` for "How this works" expanders, `show_metrics_bar()` for result metrics, `load_example()` for demo mode files, `show_sidebar_status()` for model status; no pipeline imports
- `examples/` — sample files for demo mode (`sample.pdf` copied from test fixtures)

### Key Details

- Model cached via `st.cache_resource`; model load status tracked via `st.session_state` flags shown in sidebar
- "Try with example" button uses `st.session_state` flags and `load_example()` from `ui_helpers.py`; user uploads clear the example flag
- QA images are resized so the longer dimension is 768px to stay within GPU memory limits for up to 8 pages
- PDF page count is obtained via `get_pdf_page_count()` without rendering; only selected pages are rendered via `render_pdf_pages(page_indices=...)`
- QA page validates uploads: rejects mixed PDF + image uploads and multiple PDFs

## Tests

- `tests/test_models.py` — `_load_vision_model()`, `create_granite_vision_model()`, `generate_response()` with mocked transformers classes; no model weights required
- `tests/test_utils.py` — `temp_upload()` file creation, cleanup, and exception safety; `timed()` duration measurement
- `tests/test_pdf.py` — `render_pdf_pages()` with real PDF fixture; `get_pdf_page_count()`; no model weights required
- `tests/test_qa.py` — `resize_for_qa()` dimension and aspect ratio tests; `generate_qa_response()` prompt structure, validation, and delegation to `generate_response()`; no model weights required
- `tests/test_ui_helpers.py` — `_ExampleFile` BytesIO wrapper attributes; `load_example()` file loading, name, size, seekability, and real example file validation; `show_metrics_bar()` column creation and metric rendering with mocked Streamlit; `show_sidebar_status()` model status display with mocked Streamlit

Pipeline tests import directly from `pipeline` — no Streamlit mocking needed. UI helper tests mock `streamlit` via `unittest.mock.patch`.
