# Simplified UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the UI to a three-action flow (Upload -> Ask -> Answer) for non-technical users, with an optional page override for large PDFs.

**Architecture:** Strip `streamlit_app.py` to a linear layout: upload area, question input, answer display with inline source pages. Auto-select pages. Move page selection into a collapsed expander shown only for >8 page PDFs. Remove conversation history, Q&A export, and tabbed display. Relax `generate_qa_response()` to accept 1-8 images.

**Tech Stack:** Python, Streamlit, PIL, pytest

**Spec:** `docs/superpowers/specs/2026-03-27-simplified-ui-design.md`

---

### Task 1: Update QA validation to accept 1 image

**Files:**
- Modify: `tests/test_qa.py:60-75` (validation tests)
- Modify: `pipeline/qa.py:40-41` (validation logic)

- [ ] **Step 1: Update the failing test for single-image acceptance**

In `tests/test_qa.py`, replace `test_generate_qa_response_rejects_single_image` with a test that verifies a single image is accepted:

```python
@patch("pipeline.qa.generate_response")
def test_generate_qa_response_accepts_single_image(
    mock_gen: MagicMock,
) -> None:
    mock_gen.return_value = "Answer about one page."
    images = [Image.new("RGB", (100, 100))]
    result = generate_qa_response(images, "What is this?", MagicMock(), MagicMock())
    assert result == "Answer about one page."
```

Add the `@patch` import if not already present (it is). Add this test in place of the old `test_generate_qa_response_rejects_single_image` at line 65.

- [ ] **Step 2: Update the empty-images test error message**

In `tests/test_qa.py`, update `test_generate_qa_response_rejects_empty_images` to match the new error message:

```python
def test_generate_qa_response_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response([], "What is this?", MagicMock(), MagicMock())
```

- [ ] **Step 3: Update the >8 images test error message**

In `tests/test_qa.py`, update `test_generate_qa_response_rejects_more_than_8_images` to match the new error message:

```python
def test_generate_qa_response_rejects_more_than_8_images() -> None:
    images = [Image.new("RGB", (100, 100)) for _ in range(9)]
    with pytest.raises(ValueError, match="1 to 8"):
        generate_qa_response(images, "What is this?", MagicMock(), MagicMock())
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_qa.py -v`
Expected: 2 failures — the new single-image test raises `ValueError`, and the error message tests fail on `"1 to 8"` vs `"2 to 8"`.

- [ ] **Step 5: Update the validation in generate_qa_response**

In `pipeline/qa.py`, change line 40-41 from:

```python
    if not (2 <= len(images) <= 8):
        raise ValueError(f"Expected 2 to 8 images, got {len(images)}")
```

to:

```python
    if not (1 <= len(images) <= 8):
        raise ValueError(f"Expected 1 to 8 images, got {len(images)}")
```

Also update the docstring on lines 33-38 from "Accepts 2-8 images" to "Accepts 1-8 images" and from "fewer than 2" to "fewer than 1":

```python
    """Answer a question about consecutive page images.

    Accepts 1-8 images. Each image is converted to RGB and resized so the
    longer dimension is at most 768px. All images are passed to the model
    in a single conversation turn.

    Raises ValueError if images list has fewer than 1 or more than 8 items.
    Returns empty string if the model produces no output.
    """
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_qa.py -v`
Expected: All 10 tests pass.

- [ ] **Step 7: Commit**

```bash
git add pipeline/qa.py tests/test_qa.py
git commit -m "feat: accept 1-8 images in generate_qa_response

Relaxes the minimum from 2 to 1 image to support single-page PDFs."
```

---

### Task 2: Remove format_qa_export

**Files:**
- Modify: `tests/test_ui_helpers.py:9-16,246-303` (remove import and tests)
- Modify: `ui_helpers.py:75-107` (remove function)

- [ ] **Step 1: Remove format_qa_export tests**

In `tests/test_ui_helpers.py`, remove the `format_qa_export` import from line 12 and the entire `# --- format_qa_export tests ---` section (lines 246-303). The import block becomes:

```python
from ui_helpers import (
    _ExampleFile,
    clamp_page_range,
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)
```

- [ ] **Step 2: Run tests to verify they still pass**

Run: `uv run pytest tests/test_ui_helpers.py -v`
Expected: 20 tests pass (the 6 format_qa_export tests are gone, leaving 14).

- [ ] **Step 3: Remove format_qa_export function**

In `ui_helpers.py`, remove the entire `format_qa_export` function (lines 75-107) and the `datetime` / `timezone` imports from line 10 that are now unused:

```python
from datetime import datetime, timezone
```

Remove that line. The remaining imports are:

```python
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
```

- [ ] **Step 4: Run tests to verify they still pass**

Run: `uv run pytest tests/test_ui_helpers.py -v`
Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "refactor: remove format_qa_export

Q&A export feature removed as part of UI simplification."
```

---

### Task 3: Rewrite streamlit_app.py

**Files:**
- Modify: `streamlit_app.py` (full rewrite)

- [ ] **Step 1: Rewrite streamlit_app.py**

Replace the entire contents of `streamlit_app.py` with:

```python
import streamlit as st

from pipeline import (
    create_granite_vision_model,
    generate_qa_response,
    get_pdf_page_count,
    render_pdf_pages,
    temp_upload,
    timed,
)
from ui_helpers import (
    clamp_page_range,
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)

st.set_page_config(page_title="PDF Question & Answer")

qa_model = st.cache_resource(create_granite_vision_model)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("PDF Question & Answer")
st.write("Upload a PDF, then ask a question.")

col_upload, col_example = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )
with col_example:
    if st.button("Try with example"):
        st.session_state["use_example_qa"] = True
        st.rerun()

# Resolve files: user upload takes priority over example
if uploaded_file:
    st.session_state.pop("use_example_qa", None)
elif st.session_state.get("use_example_qa"):
    uploaded_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

selected: list[int] = []

if uploaded_file:
    show_upload_preview(uploaded_file)

    with temp_upload(uploaded_file) as path:
        total_pages = get_pdf_page_count(path)
    uploaded_file.seek(0)

    # Auto-select first N pages (N = min(total_pages, 8))
    auto_end = min(total_pages, 8)
    selected = list(range(1, auto_end + 1))

    if total_pages > 8:
        # Show page override expander for large PDFs
        st.caption(f"{total_pages} pages — Pages {selected[0]}-{selected[-1]} selected")

        with st.expander("Change pages"):
            # Cache thumbnails in session state
            file_size = getattr(uploaded_file, "size", 0)
            file_key = (
                f"thumbs_{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
            )
            if file_key not in st.session_state:
                with temp_upload(uploaded_file) as thumb_path:
                    if total_pages > 50:
                        all_thumbs: list = []
                        for batch_start in range(0, total_pages, 20):
                            batch_end = min(batch_start + 20, total_pages)
                            batch_indices = list(range(batch_start, batch_end))
                            all_thumbs.extend(
                                render_pdf_pages(
                                    thumb_path, dpi=72, page_indices=batch_indices
                                )
                            )
                        st.session_state[file_key] = all_thumbs
                    else:
                        st.session_state[file_key] = render_pdf_pages(
                            thumb_path, dpi=72
                        )
                uploaded_file.seek(0)
            thumbnails = st.session_state[file_key]

            slider_key = "page_range_slider"
            slider_range = st.select_slider(
                "Select page range",
                options=list(range(1, total_pages + 1)),
                value=(1, min(total_pages, 8)),
                key=slider_key,
            )
            if not isinstance(slider_range, tuple):
                slider_range = (slider_range, slider_range)

            clamped = clamp_page_range(slider_range[0], slider_range[1], max_span=8)
            if clamped != tuple(slider_range):
                st.warning(
                    f"Maximum 8 pages — selection narrowed to pages {clamped[0]}-{clamped[1]}"
                )
                st.session_state[slider_key] = clamped
                st.rerun()

            cols_per_row = 6 if total_pages > 12 else 4
            render_thumbnail_grid(
                thumbnails, selected_range=clamped, cols_per_row=cols_per_row
            )

            selected = list(range(clamped[0], clamped[1] + 1))
            st.caption(
                f"Pages {clamped[0]}-{clamped[1]} selected ({len(selected)} pages)"
            )
    else:
        st.caption(
            f"{total_pages} page{'s' if total_pages != 1 else ''} — "
            f"All pages selected"
        )

question = st.text_area(
    "Question",
    placeholder="e.g., What is shown on these pages?",
    height=100,
)

if st.button("Answer", type="primary"):
    if not uploaded_file:
        st.warning("Upload a PDF first.")
        st.stop()
    if not selected:
        st.warning("No pages selected.")
        st.stop()
    if not question:
        st.warning("Enter a question first.")
        st.stop()

    if not st.session_state.get("model_granite_vision"):
        spinner_msg = "Loading model and generating answer..."
    else:
        spinner_msg = "Generating answer..."

    with st.spinner(spinner_msg):
        processor, model = qa_model()
        st.session_state["model_granite_vision"] = True

        with temp_upload(uploaded_file) as tmp_path:
            page_images = render_pdf_pages(
                tmp_path, page_indices=[i - 1 for i in selected]
            )

        with timed() as t:
            answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        st.session_state["last_answer"] = answer
        st.session_state["last_duration_s"] = t.duration_s
        st.session_state["last_source_pages"] = page_images
        st.session_state["last_page_numbers"] = selected

# Display result
answer = st.session_state.get("last_answer")
if answer:
    with st.container(border=True):
        st.markdown(answer)
    duration = st.session_state.get("last_duration_s")
    if duration is not None:
        st.caption(f"Generated in {duration:.2f}s")

    source_pages = st.session_state.get("last_source_pages", [])
    page_numbers = st.session_state.get("last_page_numbers", [])
    if source_pages:
        st.divider()
        cols_per_row = min(4, len(source_pages))
        for row_start in range(0, len(source_pages), cols_per_row):
            row_pages = list(
                enumerate(
                    source_pages[row_start : row_start + cols_per_row],
                    start=row_start,
                )
            )
            cols = st.columns(cols_per_row)
            for col, (idx, img) in zip(cols, row_pages):
                page_num = page_numbers[idx] if idx < len(page_numbers) else idx + 1
                col.image(img, caption=f"Page {page_num}", width="stretch")
```

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check streamlit_app.py && uv run ruff format streamlit_app.py`
Expected: No errors, file formatted.

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest -v`
Expected: All tests pass (41 total minus 6 removed format_qa_export tests minus 1 replaced single-image test = 35 tests, plus 1 new single-image test = 35 tests).

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: simplify UI to upload-ask-answer flow

Replaces tabbed display, conversation history, and Q&A export with a
linear layout. Auto-selects pages. Page override expander shown only
for PDFs with more than 8 pages. Single-page PDFs now supported."
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Project Overview**

Replace line 5:

```markdown
Streamlit web app for multipage document question answering using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Upload a PDF, select 2-8 consecutive pages via a thumbnail grid and range slider, ask a question, and get a text answer. Supports conversation history, source page verification, and Q&A export.
```

with:

```markdown
Streamlit web app for document question answering using [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Upload a PDF, ask a question, and get a text answer. Automatically selects up to 8 pages; PDFs with more than 8 pages show an optional page range picker.
```

- [ ] **Step 2: Update pipeline/qa.py description**

Replace line 51:

```markdown
- `pipeline/qa.py` -- `resize_for_qa()` image resizing; `generate_qa_response()` multi-page QA requiring 2-8 consecutive pages (delegates to `generate_response()`)
```

with:

```markdown
- `pipeline/qa.py` -- `resize_for_qa()` image resizing; `generate_qa_response()` QA accepting 1-8 page images (delegates to `generate_response()`)
```

- [ ] **Step 3: Update UI section**

Replace lines 55-57:

```markdown
- `streamlit_app.py` -- single-page app; PDF upload, thumbnail grid page selection with range slider (2-8 consecutive pages), text area question input, tabbed answer/source display with conversation history, Q&A export download
- `ui_helpers.py` -- shared UI functions: `show_upload_preview()` for PDF file info, `clamp_page_range()` for page range validation, `render_thumbnail_grid()` for page thumbnail display, `format_qa_export()` for markdown Q&A export, `load_example()` for demo mode files; no pipeline imports
- `examples/` -- sample files for demo mode
```

with:

```markdown
- `streamlit_app.py` -- single-page app; PDF upload with auto page selection, optional page range picker for >8 page PDFs, text area question input, inline answer and source page display
- `ui_helpers.py` -- shared UI functions: `show_upload_preview()` for PDF file info, `clamp_page_range()` for page range validation, `render_thumbnail_grid()` for page thumbnail display, `load_example()` for demo mode files; no pipeline imports
- `examples/` -- sample files for demo mode
```

- [ ] **Step 4: Update Key Details section**

Replace lines 61-71:

```markdown
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
```

with:

```markdown
- Only PDF uploads are supported; the file uploader accepts a single PDF
- QA accepts 1-8 page images; any PDF with at least 1 page is valid
- Pages are auto-selected on upload: first N pages where N = min(total_pages, 8)
- For PDFs with >8 pages, a collapsed "Change pages" expander shows a range slider and thumbnail grid; thumbnails rendered at 72 DPI (batched for PDFs over 50 pages), cached in `st.session_state`
- For PDFs with 8 or fewer pages, all pages are auto-selected with no picker shown
- Page images are resized so the longer dimension is 768px to stay within GPU memory limits
- PDF page count is obtained via `get_pdf_page_count()` without rendering; only selected pages are rendered at full DPI via `render_pdf_pages(page_indices=...)`
- Answer button is always clickable; validates on click (shows warnings for missing PDF/pages/question) instead of being disabled
- Single answer displayed inline with source page images below; each new answer replaces the previous one
- Model cached via `st.cache_resource`; model status shown inline via spinner messages
- "Try with example" button uses `st.session_state` flags and `load_example()` from `ui_helpers.py`; user uploads clear the example flag
```

- [ ] **Step 5: Update Tests section**

Replace lines 78-79:

```markdown
- `tests/test_qa.py` -- `resize_for_qa()` dimension and aspect ratio tests; `generate_qa_response()` prompt structure, validation (2-8 images), and delegation to `generate_response()`; no model weights required
- `tests/test_ui_helpers.py` -- `_ExampleFile` BytesIO wrapper attributes; `load_example()` file loading, name, size, seekability, and real example file validation; `show_upload_preview()` PDF preview rendering; `clamp_page_range()` range validation and clamping; `render_thumbnail_grid()` column creation, image display, page captions, and selection highlighting; `format_qa_export()` markdown structure, Q&A pairs, and timestamp format
```

with:

```markdown
- `tests/test_qa.py` -- `resize_for_qa()` dimension and aspect ratio tests; `generate_qa_response()` prompt structure, validation (1-8 images), and delegation to `generate_response()`; no model weights required
- `tests/test_ui_helpers.py` -- `_ExampleFile` BytesIO wrapper attributes; `load_example()` file loading, name, size, seekability, and real example file validation; `show_upload_preview()` PDF preview rendering; `clamp_page_range()` range validation and clamping; `render_thumbnail_grid()` column creation, image display, page captions, and selection highlighting
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for simplified UI

Reflects new upload-ask-answer flow, 1-8 image validation, removed
conversation history and Q&A export features."
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All 35 tests pass.

- [ ] **Step 2: Lint and format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors, no formatting changes needed.

- [ ] **Step 3: Type check**

Run: `uv run ty check .`
Expected: No errors (or only pre-existing warnings).
