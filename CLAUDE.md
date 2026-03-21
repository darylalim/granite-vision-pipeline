# Granite Vision Pipeline

## Project Overview

Streamlit web app that extracts and describes pictures and tables in PDF documents, and segments objects in images using natural language prompts. Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) with [SAM](https://huggingface.co/facebook/sam-vit-huge) refinement for segmentation.

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
- `docling[vlm]` — PDF parsing, table extraction, and VLM-based picture description
- `streamlit` — web UI framework
- `torch` — tensor operations and model inference for segmentation
- `transformers` — model loading (Granite Vision, SAM) for segmentation

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

Overrides (`[tool.uv]`):
- `opencv-python` replaced with `opencv-python-headless` to avoid duplicate `libavdevice` dylib conflicts with `av` on macOS

## Architecture

- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`, `get_table_content`, `create_granite_model`, `create_sam_model`, `segment`, `draw_mask`)
- `pipeline/config.py` — `create_converter()` factory, `convert()` wrapper, warning filters for upstream docling/transformers deprecations
- `pipeline/output.py` — `build_output()` produces a unified `elements` array from pictures and tables via `build_element()`; `get_description()` extracts picture descriptions from `meta` with fallback to `annotations`; `get_table_content()` extracts table markdown and structured column/row data
- `pipeline/segmentation.py` — `segment()` runs Granite Vision referring segmentation + SAM refinement; `draw_mask()` for overlay visualization; `create_granite_model()` and `create_sam_model()` factories; internal helpers for RLE parsing, mask processing, point sampling, and logit computation
- `streamlit_app.py` — PDF extraction UI; caches the converter via `st.cache_resource`, handles file upload, download, per-picture preview in expanders, and per-table preview with interactive dataframes
- `pages/segmentation.py` — segmentation UI page; image upload, text prompt, mask overlay preview, mask download; models cached via `st.cache_resource`

Key details:
- `convert()` accepts an optional `converter` parameter to reuse a cached instance, avoiding model reload on each call
- `get_description()` falls back to `pic.annotations` because docling appends `DescriptionAnnotation` after `PictureItem` construction, so the `meta` migration validator doesn't run
- PDF upload flow: upload PDF, click "Annotate", spinner, metrics (picture count, table count, duration), JSON download, per-picture preview in expanders (image + description), and per-table preview in expanders (image + interactive dataframe)
- Output JSON contains `document_info` (picture count, table count, timing) and an `elements` array with `type` discriminator (`"picture"` or `"table"`) and type-specific `content`
- Segmentation loads separate Granite Vision and SAM model instances (not shared with docling's internal model)
- Segmentation flow: upload image + text prompt, click "Segment", Granite generates coarse 24x24 mask via `"seg:"` prefix, SAM refines to pixel-accurate mask, results show overlay + downloadable mask PNG
- Adding `pages/` directory activates Streamlit multipage navigation with sidebar

## Tests

- `tests/test_config.py` — `create_converter()` factory with pipeline option verification, `convert()` with and without provided converter
- `tests/test_output.py` — `build_output()`, `build_element()`, `get_description()`, and `get_table_content()` with real Docling objects; covers pictures, tables, mixed documents, annotations fallback, meta priority
- `tests/test_segmentation.py` — `extract_segmentation()` (including malformed RLE handling), `prepare_mask()`, `sample_points()`, `compute_logits_from_mask()`, and `draw_mask()` unit tests; no model weights required

All tests import directly from `pipeline` — no Streamlit mocking needed.
