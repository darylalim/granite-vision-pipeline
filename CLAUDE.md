# Granite Vision Pipeline

## Project Overview

Streamlit web app with three capabilities:

1. **PDF Extraction** — extract and describe pictures and tables in PDF documents using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)
2. **Image Segmentation** — segment objects in images using natural language prompts, with [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) + [SAM](https://huggingface.co/facebook/sam-vit-huge) refinement
3. **DocTags Generation** — parse document images and PDFs to structured text in doctags format using [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)

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
- `docling[vlm]` — PDF parsing, table extraction, VLM-based picture description
- `pypdfium2` — PDF page rendering for doctags generation
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

- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`, `get_table_content`, `create_granite_model`, `create_sam_model`, `segment`, `draw_mask`, `create_doctags_model`, `generate_doctags`, `parse_doctags`, `export_markdown`, `render_pdf_pages`)
- `pipeline/config.py` — `create_converter()` factory, `convert()` wrapper, warning filters for upstream docling/transformers deprecations
- `pipeline/output.py` — `build_output()` produces a unified `elements` array from pictures and tables; `get_description()` extracts picture descriptions from `meta` with fallback to `annotations`; `get_table_content()` extracts table markdown and structured column/row data
- `pipeline/segmentation.py` — `segment()` runs Granite Vision referring segmentation + SAM refinement; `draw_mask()` for overlay visualization; `create_granite_model()` and `create_sam_model()` factories; internal helpers for RLE parsing, mask processing, point sampling, and logit computation
- `pipeline/doctags.py` — `create_doctags_model()` factory, `generate_doctags()` inference, `parse_doctags()` conversion to DoclingDocument, `export_markdown()` wrapper, `render_pdf_pages()` PDF-to-image rendering

### UI

- `streamlit_app.py` — PDF extraction page; file upload, annotation, per-picture and per-table preview in expanders
- `pages/segmentation.py` — segmentation page; image upload, text prompt, mask overlay preview, mask download
- `pages/doctags.py` — doctags generation page; image/PDF upload, raw doctags display, markdown preview, per-page expanders for PDFs

### Key Details

- `convert()` accepts an optional `converter` parameter to reuse a cached instance, avoiding model reload on each call
- `get_description()` falls back to `pic.annotations` because docling appends `DescriptionAnnotation` after `PictureItem` construction, so the `meta` migration validator doesn't run
- Output JSON contains `document_info` (picture count, table count, timing) and an `elements` array with `type` discriminator (`"picture"` or `"table"`) and type-specific `content`
- Segmentation loads separate Granite Vision and SAM model instances (not shared with docling's internal model)
- DocTags generation uses `ibm-granite/granite-docling-258M` loaded directly via Transformers (not Docling's VlmPipeline), with prompt `"Convert this page to docling."`
- For PDFs in doctags, pages are rendered to images via pypdfium2 at 144 DPI, then each page is processed independently
- Adding `pages/` directory activates Streamlit multipage navigation with sidebar
- All models are cached via `st.cache_resource` at the page level

## Tests

- `tests/test_config.py` — `create_converter()` factory with pipeline option verification, `convert()` with and without provided converter
- `tests/test_output.py` — `build_output()`, `build_element()`, `get_description()`, `get_table_content()` with real Docling objects; covers pictures, tables, mixed documents, annotations fallback, meta priority
- `tests/test_segmentation.py` — `extract_segmentation()`, `prepare_mask()`, `sample_points()`, `compute_logits_from_mask()`, `draw_mask()` unit tests; no model weights required
- `tests/test_doctags.py` — `render_pdf_pages()` with real PDF fixture, `parse_doctags()` with sample doctags strings, `create_doctags_model()` and `generate_doctags()` with mocked model, `export_markdown()` verification; no model weights required

All tests import directly from `pipeline` — no Streamlit mocking needed.
