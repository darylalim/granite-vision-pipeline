# Enterprise UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Streamlit UI for enterprise internal teams by adding a thumbnail grid for page selection, tabbed answer/source verification, conversation history, and downloadable Q&A exports.

**Architecture:** Progressive enhancement on the existing single-page Streamlit app. All changes are in the UI layer (`streamlit_app.py`, `ui_helpers.py`). The `pipeline/` package is unchanged. New testable logic is extracted into `ui_helpers.py` as pure functions.

**Tech Stack:** Streamlit, PIL/Pillow (for thumbnail display), pytest + unittest.mock (for testing)

**Spec:** `docs/superpowers/specs/2026-03-26-enterprise-ui-improvements-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `ui_helpers.py` | Modify | Remove `show_help`, `show_sidebar_status`, `show_metrics_bar`. Add `clamp_page_range`, `render_thumbnail_grid`, `format_qa_export`. Retain everything else. |
| `streamlit_app.py` | Modify | Restructure layout: thumbnail grid, range slider, text area, tabbed answer/source, conversation history, download button. Remove help expander, sidebar, metrics bar calls. |
| `tests/test_ui_helpers.py` | Modify | Remove tests for deleted functions. Add tests for `clamp_page_range`, `render_thumbnail_grid`, `format_qa_export`. |
| `CLAUDE.md` | Modify | Update architecture docs to reflect new UI structure. |

---

### Task 1: Remove deprecated UI helpers, update imports everywhere, and clean up tests

Remove `show_help`, `show_metrics_bar`, and `show_sidebar_status` from `ui_helpers.py` AND simultaneously clean up `streamlit_app.py` imports so that every commit leaves the app in a working state.

**Files:**
- Modify: `ui_helpers.py:26-44` (remove `show_help` and `show_metrics_bar`)
- Modify: `ui_helpers.py:62-70` (remove `show_sidebar_status`)
- Modify: `streamlit_app.py:1-17` (remove imports of deleted functions)
- Modify: `streamlit_app.py:31-40` (remove `show_help(...)` call)
- Modify: `streamlit_app.py:112` (remove `show_metrics_bar(...)` call)
- Modify: `streamlit_app.py:116-118` (remove `show_sidebar_status(...)` call)
- Modify: `tests/test_ui_helpers.py:7-14` (remove imports for deleted functions)
- Modify: `tests/test_ui_helpers.py:98-141` (remove tests for `show_metrics_bar` and `show_sidebar_status`)
- Modify: `tests/test_ui_helpers.py:180-206` (remove tests for `show_help`)

- [ ] **Step 1: Remove `show_help`, `show_metrics_bar`, and `show_sidebar_status` from `ui_helpers.py`**

Delete the three functions (lines 26-44 and 62-70). Keep everything else: `show_upload_preview`, `_ExampleFile`, `load_example`. The file should look like:

```python
"""Shared UI helper functions for Streamlit pages.

No pipeline imports — any data requiring pipeline calls is computed
at the call site and passed as a parameter.
"""

import io
from pathlib import Path

import streamlit as st


def show_upload_preview(uploaded_file: object) -> None:
    """Show preview of an uploaded PDF file (filename and size)."""
    if uploaded_file is None:
        return

    name = getattr(uploaded_file, "name", "file")
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        st.caption(f"**{name}**\n{size / 1024:.0f} KB")
    else:
        st.caption(f"**{name}**")


class _ExampleFile(io.BytesIO):
    """BytesIO wrapper with .name and .size attributes for UploadedFile compat."""

    def __init__(self, data: bytes, name: str = "", size: int = 0) -> None:
        super().__init__(data)
        self.name = name
        self.size = size


def load_example(file_path: str) -> _ExampleFile:
    """Load a file as a BytesIO wrapper that substitutes for st.UploadedFile."""
    data = Path(file_path).read_bytes()
    return _ExampleFile(data, name=Path(file_path).name, size=len(data))
```

- [ ] **Step 2: Clean up `streamlit_app.py` imports and remove calls to deleted functions**

Update the `ui_helpers` import block (lines 11-17) to remove deleted functions:

```python
from ui_helpers import (
    load_example,
    show_upload_preview,
)
```

Remove the `show_help(...)` call (lines 31-40). Remove the `show_metrics_bar(...)` call (line 112) — replace with:

```python
        st.caption(f"Generated in {t.duration_s:.2f}s")
```

Remove the `show_sidebar_status(...)` call (lines 116-118) entirely.

- [ ] **Step 3: Remove tests for deleted functions from `tests/test_ui_helpers.py`**

Remove the following test functions and their section comments:
- `test_show_metrics_bar_creates_columns` (line 102)
- `test_show_metrics_bar_empty_dict` (line 114)
- `test_show_metrics_bar_single_metric` (line 121)
- `test_show_sidebar_status_shows_models` (line 135)
- `test_show_help_renders_all_sections` (line 184)
- `test_show_help_calls_markdown_three_times` (line 199)

Also remove `show_help`, `show_metrics_bar`, `show_sidebar_status` from the import block (lines 10-13). The imports should become:

```python
from ui_helpers import (
    _ExampleFile,
    load_example,
    show_upload_preview,
)
```

- [ ] **Step 4: Run tests to verify retained tests still pass**

Run: `uv run pytest tests/test_ui_helpers.py -v`
Expected: All `_ExampleFile`, `load_example`, and `show_upload_preview` tests PASS. No tests for deleted functions.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS. Pipeline tests are unaffected.

- [ ] **Step 6: Commit**

```bash
git add ui_helpers.py streamlit_app.py tests/test_ui_helpers.py
git commit -m "refactor: remove show_help, show_metrics_bar, show_sidebar_status"
```

---

### Task 2: Add `clamp_page_range` helper with tests

**Files:**
- Modify: `ui_helpers.py` (add `clamp_page_range` function)
- Modify: `tests/test_ui_helpers.py` (add tests for `clamp_page_range`)

- [ ] **Step 1: Write failing tests for `clamp_page_range`**

Add `clamp_page_range` to the import block in `tests/test_ui_helpers.py`:

```python
from ui_helpers import (
    _ExampleFile,
    clamp_page_range,
    load_example,
    show_upload_preview,
)
```

Add test functions after the existing `show_upload_preview` tests:

```python
# --- clamp_page_range tests ---


def test_clamp_page_range_within_limit() -> None:
    assert clamp_page_range(1, 5, max_span=8) == (1, 5)


def test_clamp_page_range_exact_limit() -> None:
    assert clamp_page_range(1, 8, max_span=8) == (1, 8)


def test_clamp_page_range_exceeds_limit() -> None:
    assert clamp_page_range(1, 12, max_span=8) == (1, 8)


def test_clamp_page_range_single_page() -> None:
    assert clamp_page_range(5, 5, max_span=8) == (5, 5)


def test_clamp_page_range_exceeds_from_middle() -> None:
    assert clamp_page_range(3, 15, max_span=8) == (3, 10)


def test_clamp_page_range_custom_max_span() -> None:
    assert clamp_page_range(1, 10, max_span=4) == (1, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_helpers.py::test_clamp_page_range_within_limit -v`
Expected: FAIL with `ImportError` — `clamp_page_range` does not exist yet.

- [ ] **Step 3: Implement `clamp_page_range` in `ui_helpers.py`**

Add after the `show_upload_preview` function:

```python
def clamp_page_range(start: int, end: int, max_span: int = 8) -> tuple[int, int]:
    """Clamp a page range so it spans at most max_span pages.

    If end - start + 1 exceeds max_span, narrows the range by moving end
    to start + max_span - 1. Returns (start, end) tuple.
    """
    if end - start + 1 > max_span:
        end = start + max_span - 1
    return (start, end)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_helpers.py -k "clamp_page_range" -v`
Expected: All 6 `clamp_page_range` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: add clamp_page_range helper"
```

---

### Task 3: Add `render_thumbnail_grid` helper with tests

**Files:**
- Modify: `ui_helpers.py` (add `render_thumbnail_grid` function)
- Modify: `tests/test_ui_helpers.py` (add tests for `render_thumbnail_grid`)

- [ ] **Step 1: Write failing tests for `render_thumbnail_grid`**

Add `from PIL import Image` after `from pathlib import Path` (line 4) in `tests/test_ui_helpers.py`.

Add `render_thumbnail_grid` to the import block:

```python
from ui_helpers import (
    _ExampleFile,
    clamp_page_range,
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)
```

Add test functions after the `clamp_page_range` tests:

```python
# --- render_thumbnail_grid tests ---


@patch("ui_helpers.st")
def test_render_thumbnail_grid_creates_correct_columns(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    render_thumbnail_grid(images, selected_range=(1, 2), cols_per_row=4)

    mock_st.columns.assert_called_once_with(4)


@patch("ui_helpers.st")
def test_render_thumbnail_grid_displays_all_images(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(3)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    render_thumbnail_grid(images, selected_range=(1, 3), cols_per_row=3)

    for i, col in enumerate(mock_cols):
        container = col.container.return_value
        container.image.assert_called_once_with(images[i], use_container_width=True)


@patch("ui_helpers.st")
def test_render_thumbnail_grid_captions_show_page_numbers(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(2)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(2)]
    render_thumbnail_grid(images, selected_range=(1, 2), cols_per_row=2)

    c0 = mock_cols[0].container.return_value
    c1 = mock_cols[1].container.return_value
    c0.caption.assert_called_once_with("Page 1")
    c1.caption.assert_called_once_with("Page 2")


@patch("ui_helpers.st")
def test_render_thumbnail_grid_highlights_selected(mock_st: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    render_thumbnail_grid(images, selected_range=(2, 3), cols_per_row=4)

    # Pages 2-3 (index 1-2) should use container(border=True)
    mock_cols[1].container.assert_called_once_with(border=True)
    mock_cols[2].container.assert_called_once_with(border=True)
    # Pages 1 and 4 (index 0, 3) should use container(border=False)
    mock_cols[0].container.assert_called_once_with(border=False)
    mock_cols[3].container.assert_called_once_with(border=False)


@patch("ui_helpers.st")
def test_render_thumbnail_grid_multiple_rows(mock_st: MagicMock) -> None:
    mock_row1 = [MagicMock() for _ in range(3)]
    mock_row2 = [MagicMock() for _ in range(3)]
    mock_st.columns.side_effect = [mock_row1, mock_row2]

    images = [Image.new("RGB", (100, 100)) for _ in range(5)]
    render_thumbnail_grid(images, selected_range=(1, 5), cols_per_row=3)

    assert mock_st.columns.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_helpers.py::test_render_thumbnail_grid_creates_correct_columns -v`
Expected: FAIL with `ImportError` — `render_thumbnail_grid` does not exist yet.

- [ ] **Step 3: Implement `render_thumbnail_grid` in `ui_helpers.py`**

Add after `clamp_page_range`:

```python
def render_thumbnail_grid(
    images: list,
    selected_range: tuple[int, int],
    cols_per_row: int = 4,
) -> None:
    """Render page thumbnails in a grid with selected pages highlighted.

    All pages are rendered inside a container. Selected pages use
    border=True for visual highlighting; unselected use border=False.

    Args:
        images: List of PIL Images, one per page.
        selected_range: 1-based inclusive (start, end) of selected pages.
        cols_per_row: Number of columns per row.
    """
    sel_start, sel_end = selected_range
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (col, img) in enumerate(zip(cols, row_images)):
            page_num = row_start + i + 1  # 1-based
            is_selected = sel_start <= page_num <= sel_end
            container = col.container(border=is_selected)
            container.image(img, use_container_width=True)
            container.caption(f"Page {page_num}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_helpers.py -k "render_thumbnail_grid" -v`
Expected: All 5 `render_thumbnail_grid` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: add render_thumbnail_grid helper"
```

---

### Task 4: Add `format_qa_export` helper with tests

**Files:**
- Modify: `ui_helpers.py` (add `format_qa_export` function)
- Modify: `tests/test_ui_helpers.py` (add tests for `format_qa_export`)

- [ ] **Step 1: Write failing tests for `format_qa_export`**

Add `format_qa_export` to the import block in `tests/test_ui_helpers.py`:

```python
from ui_helpers import (
    _ExampleFile,
    clamp_page_range,
    format_qa_export,
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)
```

Add test functions after the `render_thumbnail_grid` tests:

```python
# --- format_qa_export tests ---


def test_format_qa_export_header() -> None:
    result = format_qa_export(
        file_name="report.pdf",
        page_range=(3, 6),
        qa_pairs=[{"question": "Q1?", "answer": "A1."}],
    )
    assert "# QA Export" in result
    assert "report.pdf" in result
    assert "pages 3-6" in result


def test_format_qa_export_contains_qa_pairs() -> None:
    pairs = [
        {"question": "What is X?", "answer": "X is Y."},
        {"question": "How about Z?", "answer": "Z is W."},
    ]
    result = format_qa_export("doc.pdf", (1, 2), pairs)
    assert "**Q:** What is X?" in result
    assert "**A:** X is Y." in result
    assert "**Q:** How about Z?" in result
    assert "**A:** Z is W." in result


def test_format_qa_export_contains_timestamp() -> None:
    result = format_qa_export("doc.pdf", (1, 2), [{"question": "Q?", "answer": "A."}])
    assert "Generated:" in result
    # UTC ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
    import re

    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", result)


def test_format_qa_export_empty_pairs() -> None:
    result = format_qa_export("doc.pdf", (1, 2), [])
    assert "# QA Export" in result
    assert "## Q&A" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_helpers.py::test_format_qa_export_header -v`
Expected: FAIL with `ImportError` — `format_qa_export` does not exist yet.

- [ ] **Step 3: Implement `format_qa_export` in `ui_helpers.py`**

Add the `datetime` import at the top of the file alongside existing imports:

```python
from datetime import datetime, timezone
```

Add the function after `render_thumbnail_grid`:

```python
def format_qa_export(
    file_name: str,
    page_range: tuple[int, int],
    qa_pairs: list[dict[str, str]],
) -> str:
    """Format Q&A session as a markdown string for download.

    Args:
        file_name: Name of the source PDF file.
        page_range: 1-based inclusive (start, end) page range.
        qa_pairs: List of dicts with "question" and "answer" keys.

    Returns:
        Markdown string with header, timestamp, and Q&A pairs.
    """
    start, end = page_range
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# QA Export — {file_name} (pages {start}-{end})",
        f"Generated: {timestamp}",
        "",
        "## Q&A",
    ]

    for pair in qa_pairs:
        lines.append("")
        lines.append(f"**Q:** {pair['question']}")
        lines.append(f"**A:** {pair['answer']}")

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_helpers.py -k "format_qa_export" -v`
Expected: All 4 `format_qa_export` tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: add format_qa_export helper"
```

---

### Task 5: Restructure `streamlit_app.py` — upload section

Update imports to use new helpers and streamline the upload flow. By this point, Task 1 already removed calls to deleted functions and Task 2-4 added the new helpers.

**Files:**
- Modify: `streamlit_app.py` (imports and upload section)

- [ ] **Step 1: Update imports in `streamlit_app.py`**

Replace the `ui_helpers` import block with:

```python
from ui_helpers import (
    clamp_page_range,
    format_qa_export,
    load_example,
    render_thumbnail_grid,
    show_upload_preview,
)
```

- [ ] **Step 2: Add page count display after upload preview**

After the existing `show_upload_preview(uploaded_file)` call and the `get_pdf_page_count` block, add a page count caption. The upload section should read:

```python
if uploaded_file:
    show_upload_preview(uploaded_file)

    with temp_upload(uploaded_file) as path:
        total_pages = get_pdf_page_count(path)
    uploaded_file.seek(0)

    st.caption(f"{total_pages} pages")
```

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "refactor: update imports and add page count to upload section"
```

---

### Task 6: Restructure `streamlit_app.py` — thumbnail grid and range slider

Replace the `st.number_input` page selectors with the thumbnail grid and range slider.

**Files:**
- Modify: `streamlit_app.py` (page selection section)

- [ ] **Step 1: Replace number inputs with thumbnail grid and range slider**

Replace the page selection block (from `if total_pages < 2:` through `selected = list(...)`) with:

```python
    if total_pages < 2:
        st.error("PDF must have at least 2 pages.")
    else:
        # Render thumbnails at low DPI, cache in session state
        file_size = getattr(uploaded_file, "size", 0)
        file_key = f"thumbs_{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
        if file_key not in st.session_state:
            with temp_upload(uploaded_file) as thumb_path:
                # Batch render for large PDFs to avoid blocking
                if total_pages > 50:
                    all_thumbs: list = []
                    for batch_start in range(0, total_pages, 20):
                        batch_end = min(batch_start + 20, total_pages)
                        batch_indices = list(range(batch_start, batch_end))
                        all_thumbs.extend(
                            render_pdf_pages(thumb_path, dpi=72, page_indices=batch_indices)
                        )
                    st.session_state[file_key] = all_thumbs
                else:
                    st.session_state[file_key] = render_pdf_pages(thumb_path, dpi=72)
            uploaded_file.seek(0)
        thumbnails = st.session_state[file_key]

        # Dynamic columns: 4 for small PDFs, 6 for larger ones
        cols_per_row = 6 if total_pages > 12 else 4

        # Range slider for consecutive page selection
        slider_key = "page_range_slider"
        if total_pages == 2:
            slider_range = (1, 2)
        else:
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (1, min(total_pages, 8))
            slider_range = st.select_slider(
                "Select page range",
                options=list(range(1, total_pages + 1)),
                key=slider_key,
            )

        # Clamp to max 8 pages
        clamped = clamp_page_range(slider_range[0], slider_range[1], max_span=8)
        if clamped != tuple(slider_range):
            st.warning(
                f"Maximum 8 pages — selection narrowed to pages {clamped[0]}-{clamped[1]}"
            )
            st.session_state[slider_key] = clamped
            st.rerun()

        # Display thumbnail grid
        render_thumbnail_grid(thumbnails, selected_range=clamped, cols_per_row=cols_per_row)
        num_selected = clamped[1] - clamped[0] + 1
        st.caption(f"Pages {clamped[0]}-{clamped[1]} selected ({num_selected} pages)")

        selected = list(range(clamped[0], clamped[1] + 1))
```

- [ ] **Step 2: Clear conversation history when page selection changes**

Add after the `selected = ...` line:

```python
        # Reset conversation history if selection changed
        prev_sel = st.session_state.get("prev_selected")
        if prev_sel != selected:
            st.session_state["qa_history"] = []
            st.session_state["source_pages"] = []
            st.session_state["prev_selected"] = selected
```

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add thumbnail grid and range slider for page selection"
```

---

### Task 7: Restructure `streamlit_app.py` — question input and answer display

Replace `st.text_input` with `st.text_area`, add tabbed answer/source display, conversation history, and download button.

**Files:**
- Modify: `streamlit_app.py` (question and answer section)

- [ ] **Step 1: Replace text input with text area**

Replace the `question = st.text_input(...)` line with:

```python
question = st.text_area(
    "Question",
    placeholder="e.g., What is shown on these pages?",
    height=100,
)
st.caption("Ctrl+Enter to submit")
```

- [ ] **Step 2: Replace the answer button and display logic**

Replace the entire answer block (from `has_input = ...` through the end of the file) with:

```python
has_input = bool(uploaded_file) and len(selected) >= 2 and bool(question)

if st.button("Answer", type="primary", disabled=not has_input):
    assert uploaded_file is not None

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

        # Store source pages (pre-resize) for verification tab
        st.session_state["source_pages"] = page_images

        with timed() as t:
            answer = generate_qa_response(page_images, question, processor, model)

    if not answer:
        st.warning("Model produced no output.")
    else:
        # Append to conversation history
        history = st.session_state.get("qa_history", [])
        history.append({"question": question, "answer": answer})
        st.session_state["qa_history"] = history
        st.session_state["last_duration"] = t.duration_s

# Display results if there is history
history = st.session_state.get("qa_history", [])
source_pages = st.session_state.get("source_pages", [])

if history:
    tab_answer, tab_source = st.tabs(["Answer", "Source Pages"])

    with tab_answer:
        for entry in history:
            st.markdown(f"**Q:** {entry['question']}")
            with st.container(border=True):
                st.markdown(entry["answer"])
            st.divider()

        duration = st.session_state.get("last_duration")
        if duration is not None:
            st.caption(f"Generated in {duration:.2f}s")

        # Download button
        if uploaded_file and selected:
            file_name = getattr(uploaded_file, "name", "document.pdf")
            page_range = (selected[0], selected[-1])
            export_md = format_qa_export(file_name, page_range, history)
            st.download_button(
                "Download Q&A",
                data=export_md,
                file_name="qa_export.md",
                mime="text/markdown",
            )

    with tab_source:
        if source_pages:
            src_cols_per_row = min(4, len(source_pages))
            for row_start in range(0, len(source_pages), src_cols_per_row):
                row_pages = list(
                    enumerate(
                        source_pages[row_start : row_start + src_cols_per_row],
                        start=row_start,
                    )
                )
                cols = st.columns(src_cols_per_row)
                for col, (idx, img) in zip(cols, row_pages):
                    page_num = selected[idx] if idx < len(selected) else idx + 1
                    col.image(img, caption=f"Page {page_num}", use_container_width=True)
        else:
            st.info("Source pages will appear here after generating an answer.")
```

- [ ] **Step 3: Clear history when a new file is uploaded**

In the file resolution block near the top, after `st.session_state.pop("use_example_qa", None)`, add:

```python
    # Reset history for new upload
    st.session_state["qa_history"] = []
    st.session_state["source_pages"] = []
```

- [ ] **Step 4: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS. Pipeline tests unaffected.

- [ ] **Step 6: Run type check**

Run: `uv run ty check .`
Expected: No new errors.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add tabbed answer/source display, conversation history, and download"
```

---

### Task 8: Update `CLAUDE.md` documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Architecture > UI section**

In `CLAUDE.md`, update the UI description to reflect the new structure:

- `streamlit_app.py` — single-page app; PDF upload, thumbnail grid page selection with range slider (2-8 consecutive pages), text area question input, tabbed answer/source display with conversation history, Q&A export download
- `ui_helpers.py` — shared UI functions: `show_upload_preview()` for PDF file info, `clamp_page_range()` for page range validation, `render_thumbnail_grid()` for page thumbnail display, `format_qa_export()` for markdown Q&A export, `load_example()` for demo mode files; no pipeline imports

- [ ] **Step 2: Update the Architecture > Key Details section**

Update key details to reflect:
- Page selection uses a range slider with thumbnail grid (replaces start page + number of pages inputs)
- Answer is displayed in tabbed view: Answer tab (with conversation history) and Source Pages tab
- Conversation history stored in `st.session_state`; resets on new upload or page selection change
- Q&A sessions can be exported as markdown via download button
- Remove references to `show_help`, `show_sidebar_status`, `show_metrics_bar`
- Model status shown inline via spinner messages (no sidebar)

- [ ] **Step 3: Update the Tests section**

Update `tests/test_ui_helpers.py` description:
- `tests/test_ui_helpers.py` — `_ExampleFile` BytesIO wrapper attributes; `load_example()` file loading, name, size, seekability, and real example file validation; `show_upload_preview()` PDF preview rendering; `clamp_page_range()` range validation and clamping; `render_thumbnail_grid()` column creation, image display, page captions, and selection highlighting; `format_qa_export()` markdown structure, Q&A pairs, and timestamp format

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for enterprise UI improvements"
```

---

### Task 9: Final verification and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full lint, format, and type check**

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
```

Expected: No errors.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 3: Verify the app runs**

Run: `uv run streamlit run streamlit_app.py`
Manual checks:
- App loads without errors
- Upload section shows file preview and page count
- Thumbnail grid renders after upload
- Range slider selects consecutive pages
- Selected thumbnails are highlighted with borders
- Clamping to 8 pages works (warning shown, slider updates)
- Question text area accepts multi-line input
- Answer appears in tabbed view
- Source Pages tab shows pre-resize page images
- Conversation history accumulates within Answer tab
- Download button exports markdown with correct format
- New upload resets history
- Changing page selection resets history

- [ ] **Step 4: Commit any cleanup**

```bash
git add -A
git commit -m "chore: final cleanup for enterprise UI improvements"
```
