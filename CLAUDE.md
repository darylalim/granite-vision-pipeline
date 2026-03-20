# Granite Vision Pipeline

## Project Overview

Streamlit web app that describes pictures in PDF documents using a local vision language model: [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b).

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
- `docling[vlm]` — PDF parsing and VLM-based picture description
- `streamlit` — web UI framework

Dev (`[dependency-groups] dev`):
- `pytest` — testing
- `ruff` — linting and formatting
- `ty` — type checking

Overrides (`[tool.uv]`):
- `opencv-python` replaced with `opencv-python-headless` to avoid duplicate `libavdevice` dylib conflicts with `av` on macOS

## Architecture

- `pipeline/__init__.py` — re-exports public API (`convert`, `create_converter`, `build_output`, `get_description`)
- `pipeline/config.py` — `create_converter()` factory, `convert()` wrapper, warning filters for upstream docling/transformers deprecations
- `pipeline/output.py` — `build_output()` pure function that builds output dict from a `DoclingDocument` and duration; `get_description()` reads from `pic.meta.description` with fallback to `pic.annotations`
- `streamlit_app.py` — UI only; caches the converter via `st.cache_resource`, passes it to `convert()`, handles file upload, download, and in-app picture preview with expanders

Key details:
- `convert()` accepts an optional `converter` parameter to reuse a cached instance, avoiding model reload on each call
- `get_description()` falls back to `pic.annotations` because docling appends `DescriptionAnnotation` after `PictureItem` construction, so the `meta` migration validator doesn't run
- Upload flow: upload PDF, click "Annotate", spinner, metrics (picture count, duration), JSON download, and per-picture preview in expanders (image + description)
- Output JSON contains `document_info` (count, timing) and a `pictures` array (reference, caption, description)

## Tests

- `tests/test_config.py` — `create_converter()` factory, `convert()` with and without provided converter
- `tests/test_output.py` — `build_output()` and `get_description()` with real Docling objects, annotations fallback, meta priority over annotations

All tests import directly from `pipeline` — no Streamlit mocking needed.
