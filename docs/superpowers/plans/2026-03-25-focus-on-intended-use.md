# Focus App on Granite Vision Intended Use — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the app to PDF Extraction and Multipage QA only, removing Image Segmentation, Document Parsing, and Document Search along with all their code, tests, dependencies, and UI.

**Architecture:** Relocate `render_pdf_pages()` and `get_pdf_page_count()` from `pipeline/doctags.py` to a new `pipeline/pdf.py` before deleting doctags. Delete all pipeline modules, pages, tests, and example files for removed features. Strip indexing from the extraction page. Update dependencies, docs, and config.

**Tech Stack:** Python, Streamlit, pypdfium2, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-focus-on-intended-use-design.md`

---

### Task 1: Relocate PDF rendering functions to `pipeline/pdf.py`

**Files:**
- Create: `pipeline/pdf.py`
- Create: `tests/test_pdf.py`

- [ ] **Step 1: Create `pipeline/pdf.py`**

Copy `render_pdf_pages()` and `get_pdf_page_count()` from `pipeline/doctags.py` into a new module. These are pure pypdfium2 wrappers.

```python
"""PDF rendering utilities via pypdfium2."""

import pypdfium2
from PIL import Image


def render_pdf_pages(
    pdf_path: str,
    dpi: int = 144,
    page_indices: list[int] | None = None,
) -> list[Image.Image]:
    """Render pages of a PDF to PIL RGB Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Default 144.
        page_indices: Zero-based page indices to render. Default None renders all.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        indices = page_indices if page_indices is not None else list(range(len(pdf)))
        pages: list[Image.Image] = []
        for i in indices:
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil().convert("RGB")
            pages.append(pil_image)
        return pages
    finally:
        pdf.close()


def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering."""
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()
```

- [ ] **Step 2: Create `tests/test_pdf.py`**

Relocate the 6 PDF rendering tests from `tests/test_doctags.py`, updating the import path from `pipeline.doctags` to `pipeline.pdf`.

```python
"""Tests for the PDF rendering module."""

from pathlib import Path

from PIL import Image

from pipeline.pdf import get_pdf_page_count, render_pdf_pages

TEST_PDF = str(Path(__file__).parent / "data" / "pdf" / "test_pictures.pdf")


def test_render_pdf_pages_returns_list_of_images() -> None:
    pages = render_pdf_pages(TEST_PDF)
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, Image.Image)


def test_render_pdf_pages_images_have_nonzero_dimensions() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        w, h = page.size
        assert w > 0
        assert h > 0


def test_render_pdf_pages_images_are_rgb() -> None:
    pages = render_pdf_pages(TEST_PDF)
    for page in pages:
        assert page.mode == "RGB"


def test_render_pdf_pages_with_page_indices() -> None:
    all_pages = render_pdf_pages(TEST_PDF)
    first_page = render_pdf_pages(TEST_PDF, page_indices=[0])
    assert len(first_page) == 1
    assert first_page[0].size == all_pages[0].size


def test_get_pdf_page_count_returns_correct_count() -> None:
    all_pages = render_pdf_pages(TEST_PDF)
    count = get_pdf_page_count(TEST_PDF)
    assert count == len(all_pages)


def test_get_pdf_page_count_positive() -> None:
    assert get_pdf_page_count(TEST_PDF) > 0
```

- [ ] **Step 3: Run new tests**

Run: `uv run pytest tests/test_pdf.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add pipeline/pdf.py tests/test_pdf.py
git commit -m "refactor: relocate PDF rendering functions to pipeline/pdf.py"
```

---

### Task 2: Delete removed feature files

**Files:**
- Delete: `pipeline/segmentation.py`
- Delete: `pipeline/doctags.py`
- Delete: `pipeline/search.py`
- Delete: `pages/segmentation.py`
- Delete: `pages/doctags.py`
- Delete: `pages/search.py`
- Delete: `streamlit_home.py`
- Delete: `tests/test_segmentation.py`
- Delete: `tests/test_doctags.py`
- Delete: `tests/test_search.py`
- Delete: `examples/sample.jpg`

- [ ] **Step 1: Delete all files for removed features**

```bash
git rm pipeline/segmentation.py pipeline/doctags.py pipeline/search.py
git rm pages/segmentation.py pages/doctags.py pages/search.py
git rm streamlit_home.py
git rm tests/test_segmentation.py tests/test_doctags.py tests/test_search.py
git rm examples/sample.jpg
```

- [ ] **Step 2: Commit**

```bash
git commit -m "remove: delete segmentation, doctags, search, and landing page files"
```

---

### Task 3: Update `pipeline/__init__.py`

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Rewrite `pipeline/__init__.py`**

Replace the entire file with only the remaining re-exports:

```python
from pipeline.config import convert, create_converter
from pipeline.models import create_granite_vision_model, generate_response
from pipeline.output import build_output, get_description, get_table_content
from pipeline.pdf import get_pdf_page_count, render_pdf_pages
from pipeline.qa import generate_qa_response, resize_for_qa
from pipeline.utils import timed, temp_upload

__all__ = [
    "build_output",
    "convert",
    "create_converter",
    "create_granite_vision_model",
    "generate_qa_response",
    "generate_response",
    "get_description",
    "get_pdf_page_count",
    "get_table_content",
    "render_pdf_pages",
    "resize_for_qa",
    "temp_upload",
    "timed",
]
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from pipeline import render_pdf_pages, get_pdf_page_count, create_granite_vision_model; print('OK')"`
Expected: Prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add pipeline/__init__.py
git commit -m "refactor: update pipeline re-exports for two-feature app"
```

---

### Task 4: Update `pipeline/models.py`

**Files:**
- Modify: `pipeline/models.py`

- [ ] **Step 1: Remove `create_doctags_model()` and update docstrings**

The file should become:

```python
"""Shared model loading and generation helpers."""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def _load_vision_model(
    repo_id: str, device: str | None = None
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load an AutoProcessor and AutoModelForVision2Seq from a HuggingFace repo.

    When device is None, auto-detects: CUDA if available, else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(repo_id)
    model = AutoModelForVision2Seq.from_pretrained(repo_id).to(device)
    return processor, model


def create_granite_vision_model(
    device: str | None = None,
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    """Load Granite Vision 3.3 2B.

    Shared across QA and PDF extraction (via Docling).
    """
    return _load_vision_model("ibm-granite/granite-vision-3.3-2b", device)


def generate_response(
    conversation: list[dict],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    max_new_tokens: int = 1024,
) -> str:
    """Generate a response from a conversation using apply_chat_template.

    Handles the common pattern: tokenize conversation, generate, trim
    input tokens from output, decode. Returns decoded string, or empty
    string if the model produces no new tokens.
    """
    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(  # type: ignore[operator]
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = output[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(trimmed[0], skip_special_tokens=True)  # type: ignore[operator]
    return decoded
```

- [ ] **Step 2: Remove `test_create_doctags_model_uses_correct_repo` from `tests/test_models.py`**

Delete lines 43-58 (the entire `test_create_doctags_model_uses_correct_repo` function and its decorators).

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 tests PASS (down from 5).

- [ ] **Step 4: Commit**

```bash
git add pipeline/models.py tests/test_models.py
git commit -m "refactor: remove create_doctags_model and update docstrings"
```

---

### Task 5: Update `pages/extraction.py` — remove indexing

**Files:**
- Modify: `pages/extraction.py`

- [ ] **Step 1: Remove indexing imports, caches, and logic**

The file should become:

```python
import json

import streamlit as st
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument

from pipeline import (
    build_output,
    convert,
    create_converter,
    get_description,
    temp_upload,
    timed,
)
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

converter = st.cache_resource(create_converter)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("PDF Extraction")
st.write(
    "Extract and describe pictures and tables in PDF documents using IBM Granite Vision."
)

show_help(
    supported_formats="PDF",
    description=(
        "Uploads a PDF and uses Docling to extract pictures and tables. "
        "Pictures are described using Granite Vision. Tables are parsed into "
        "structured data. Results are available as a JSON download."
    ),
    model_info="[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) via Docling",
)

col_upload, col_example = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader("Upload file", type=["pdf"])
with col_example:
    if st.button("Try with example"):
        st.session_state["use_example_extraction"] = True
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_extraction", None)
elif st.session_state.get("use_example_extraction"):
    active_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

if active_file:
    show_upload_preview(active_file)


def _render_elements(doc: DoclingDocument) -> None:
    """Display extracted pictures and tables in expanders."""
    for idx, pic in enumerate(doc.pictures, 1):
        with st.expander(f"Picture {idx}", expanded=idx == 1):
            col_img, col_desc = st.columns(2)
            image = pic.get_image(doc)
            if image:
                col_img.image(image)
            caption = pic.caption_text(doc=doc)
            if caption:
                col_img.caption(caption)
            desc = get_description(pic)
            if desc:
                col_desc.markdown(desc["text"])
            else:
                col_desc.write("No description available.")

    for idx, table in enumerate(doc.tables, 1):
        with st.expander(
            f"Table {idx}",
            expanded=len(doc.pictures) == 0 and idx == 1,
        ):
            col_img, col_data = st.columns(2)
            image = table.get_image(doc)
            if image:
                col_img.image(image)
            caption = table.caption_text(doc=doc)
            if caption:
                col_img.caption(caption)
            df = table.export_to_dataframe(doc=doc)
            if not df.empty:
                col_data.dataframe(df)
            else:
                col_data.write("Empty table.")


if st.button("Annotate", type="primary", disabled=not active_file):
    assert active_file is not None
    file_name = getattr(active_file, "name", "document.pdf")
    try:
        with temp_upload(active_file) as tmp_path:
            with st.spinner(
                "Extracting content... This may take a few minutes for large documents."
            ):
                with timed() as t:
                    doc = convert(tmp_path, converter=converter())
                    st.session_state["model_docling"] = True

        st.success("Done.")

        show_metrics_bar(
            {
                "Pictures": len(doc.pictures),
                "Tables": len(doc.tables),
                "Duration (s)": f"{t.duration_s:.2f}",
            }
        )

        output = build_output(doc, t.duration_s)

        st.download_button(
            label="Download JSON",
            data=json.dumps(output, indent=2),
            file_name=f"{file_name}_annotations.json",
            mime="application/json",
        )

        _render_elements(doc)

    except ConversionError as e:
        st.error(str(e))

show_sidebar_status(
    models={"Docling": st.session_state.get("model_docling", False)},
)
```

Changes from original:
- Removed imports: `cast`, `create_embedding_model`, `get_collection`, `index_elements`
- Removed `embedding_model` and `collection` cache lines
- Removed the `try/except` block calling `index_elements` (lines 136-151)
- Removed `index_count` logic (lines 158-161)
- Removed `"Embedding"` from sidebar status and `index_count` parameter
- Updated `show_help()` description — removed "automatically indexed for the Document Search page"

- [ ] **Step 2: Commit**

```bash
git add pages/extraction.py
git commit -m "refactor: remove search indexing from extraction page"
```

---

### Task 6: Update `streamlit_app.py`

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Rewrite to two pages only**

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

- [ ] **Step 2: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: update navigation to two-page app"
```

---

### Task 7: Update `ui_helpers.py` and its tests

**Files:**
- Modify: `ui_helpers.py:85-98`
- Modify: `tests/test_ui_helpers.py:92-100,139-172`

- [ ] **Step 1: Remove `index_count` from `show_sidebar_status()`**

In `ui_helpers.py`, replace the `show_sidebar_status` function (lines 85-98) with:

```python
def show_sidebar_status(
    models: dict[str, bool],
) -> None:
    """Show model status in the sidebar."""
    with st.sidebar:
        st.markdown("**Models**")
        for name, loaded in models.items():
            status = "Loaded" if loaded else "Not loaded"
            st.text(f"{name}: {status}")
```

- [ ] **Step 2: Update `tests/test_ui_helpers.py`**

Update `test_load_example_with_real_example_file` (lines 92-100) — replace with:

```python
def test_load_example_with_real_example_file() -> None:
    """Test with actual example files in the repo."""
    result = load_example("examples/sample.pdf")

    assert result.name == "sample.pdf"
    assert result.size > 0
```

Remove `test_show_sidebar_status_shows_index_count` (lines 150-155), `test_show_sidebar_status_singular_document` (lines 158-162), and `test_show_sidebar_status_no_index_count` (lines 165-172).

Update `test_show_sidebar_status_shows_models` (line 141) — change `"SAM"` to `"Docling"` for clarity:

```python
@patch("ui_helpers.st")
def test_show_sidebar_status_shows_models(mock_st: MagicMock) -> None:
    show_sidebar_status({"Granite Vision": True, "Docling": False})

    mock_st.markdown.assert_any_call("**Models**")
    text_calls = mock_st.text.call_args_list
    assert call("Granite Vision: Loaded") in text_calls
    assert call("Docling: Not loaded") in text_calls
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_ui_helpers.py -v`
Expected: 13 tests PASS (down from 16 — removed 3 index_count tests).

- [ ] **Step 4: Commit**

```bash
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "refactor: remove index_count from show_sidebar_status"
```

---

### Task 8: Update dependencies and config

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Update `pyproject.toml`**

Remove `chromadb` and `sentence-transformers` from dependencies. Update description.

Change `description` to:
```
"Extract and describe pictures and tables in PDFs, and answer questions across multiple document pages"
```

Change `dependencies` to:
```toml
dependencies = [
    "docling[vlm]",
    "pypdfium2",
    "streamlit",
    "torch",
    "transformers",
]
```

- [ ] **Step 2: Update `.gitignore`**

Remove lines 17-18 (the `# Vector database` comment and `.chroma/` entry).

- [ ] **Step 3: Regenerate lock file**

Run: `uv lock`
Expected: Lock file regenerated without `chromadb` or `sentence-transformers`.

- [ ] **Step 4: Check if opencv override is still needed**

Run: `uv tree | grep -i opencv`
If no opencv packages appear, remove the `override-dependencies` and `constraint-dependencies` lines from `[tool.uv]` in `pyproject.toml` and re-run `uv lock`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "chore: remove chromadb and sentence-transformers dependencies"
```

---

### Task 9: Run full test suite and lint

- [ ] **Step 1: Run linter**

Run: `uv run ruff check .`
Expected: No errors.

- [ ] **Step 2: Run formatter**

Run: `uv run ruff format --check .`
Expected: All files already formatted.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All remaining tests pass. Test count should be:
- `test_config.py`: 5 tests
- `test_output.py`: 20 tests
- `test_models.py`: 4 tests
- `test_utils.py`: 4 tests
- `test_pdf.py`: 6 tests
- `test_qa.py`: 11 tests
- `test_ui_helpers.py`: 13 tests
- **Total: 63 tests**

- [ ] **Step 4: Fix any issues and commit if needed**

If lint or tests fail, fix and commit:
```bash
git add -u
git commit -m "fix: address lint/test issues after feature removal"
```

---

### Task 10: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Rewrite to reflect two-feature app. Key changes:
- Project Overview: only PDF Extraction and Multipage QA
- Dependencies: remove `chromadb`, `sentence-transformers`; update `transformers` description to just "Granite Vision model loading"
- Architecture Pipeline section: remove `pipeline/segmentation.py`, `pipeline/doctags.py`, `pipeline/search.py`; add `pipeline/pdf.py` — `render_pdf_pages()` PDF-to-image rendering; `get_pdf_page_count()` for page count without rendering
- Architecture UI section: remove `streamlit_home.py`, `pages/segmentation.py`, `pages/doctags.py`, `pages/search.py`; remove "auto-indexes for search" from extraction description; remove `sample.jpg` from examples description
- `pipeline/models.py` description: remove `create_doctags_model()`; update `create_granite_vision_model()` description to "shared across QA and PDF extraction"
- Key Details: remove all segmentation, doctags, search, SAM, ChromaDB, embedding references
- Tests section: remove `test_segmentation.py`, `test_doctags.py`, `test_search.py`; add `test_pdf.py`; update `test_models.py` description to remove `create_doctags_model()`; update `test_ui_helpers.py` description to remove "index count formatting, and singular/plural handling"

- [ ] **Step 2: Update `README.md`**

Rewrite to reflect two-feature app:
- Title section: two capabilities only
- "Powered by" line: only `granite-vision-3.3-2b`
- Features section: only PDF Extraction and Multipage QA; remove "indexed for search" from extraction
- Project Structure: remove deleted files; add `pipeline/pdf.py` and `tests/test_pdf.py`

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README.md for two-feature app"
```
