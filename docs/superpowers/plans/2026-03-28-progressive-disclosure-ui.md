# Progressive Disclosure UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the Streamlit UI for non-technical users via progressive disclosure — hide controls until relevant, show one clear action at a time.

**Architecture:** Two files change: `ui_helpers.py` gets an updated `show_upload_preview()` that merges file info and page range into one line, and `streamlit_app.py` is restructured to hide the question bar until a file is uploaded, replace the expander/thumbnail page picker with a toggled slider-only container, and collapse source pages in the answer card. Pipeline code is untouched.

**Tech Stack:** Streamlit (>=1.55.0), Python 3

**Spec:** `docs/superpowers/specs/2026-03-28-progressive-disclosure-ui-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ui_helpers.py` | Modify | Update `show_upload_preview()` signature and output |
| `tests/test_ui_helpers.py` | Modify | Update `show_upload_preview` tests for new signature |
| `streamlit_app.py` | Modify | Restructure layout, page picker, answer card, empty state |

---

### Task 1: Update `show_upload_preview()` to include page info

The current function only shows filename and size. The spec merges this with the page range into a single info line: `filename — size · Pages X–Y of Z` (for >8 pages) or `filename — size · N pages` (for ≤8 pages).

**Files:**
- Modify: `tests/test_ui_helpers.py`
- Modify: `ui_helpers.py`

- [ ] **Step 1: Write failing tests for the new `show_upload_preview` signature**

Replace the three existing `show_upload_preview` tests with tests for the new signature `show_upload_preview(uploaded_file, total_pages, selected)`. The function now takes page context and produces a richer caption.

```python
# In tests/test_ui_helpers.py — replace the three existing show_upload_preview tests with:

@patch("ui_helpers.st")
def test_show_upload_preview_with_pages_over_8(mock_st: MagicMock) -> None:
    buf = io.BytesIO(b"fake pdf")
    buf.name = "doc.pdf"
    buf.size = 2048  # type: ignore[attr-defined]

    show_upload_preview(buf, total_pages=32, selected=[1, 2, 3, 4, 5, 6, 7, 8])

    mock_st.caption.assert_called_once()
    caption_arg = mock_st.caption.call_args[0][0]
    assert "doc.pdf" in caption_arg
    assert "2 KB" in caption_arg
    assert "Pages 1–8 of 32" in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_with_pages_8_or_fewer(mock_st: MagicMock) -> None:
    buf = io.BytesIO(b"fake pdf")
    buf.name = "report.pdf"
    buf.size = 1024  # type: ignore[attr-defined]

    show_upload_preview(buf, total_pages=3, selected=[1, 2, 3])

    mock_st.caption.assert_called_once()
    caption_arg = mock_st.caption.call_args[0][0]
    assert "report.pdf" in caption_arg
    assert "3 pages" in caption_arg
    # Should NOT contain "of" since all pages are selected
    assert " of " not in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_single_page(mock_st: MagicMock) -> None:
    buf = io.BytesIO(b"fake pdf")
    buf.name = "one.pdf"
    buf.size = 512  # type: ignore[attr-defined]

    show_upload_preview(buf, total_pages=1, selected=[1])

    caption_arg = mock_st.caption.call_args[0][0]
    assert "one.pdf" in caption_arg
    assert "1 page" in caption_arg


@patch("ui_helpers.st")
def test_show_upload_preview_none(mock_st: MagicMock) -> None:
    show_upload_preview(None, total_pages=0, selected=[])

    mock_st.caption.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_helpers.py::test_show_upload_preview_with_pages_over_8 tests/test_ui_helpers.py::test_show_upload_preview_with_pages_8_or_fewer tests/test_ui_helpers.py::test_show_upload_preview_single_page tests/test_ui_helpers.py::test_show_upload_preview_none -v`

Expected: FAIL — `show_upload_preview()` does not accept `total_pages` or `selected` parameters.

- [ ] **Step 3: Update `show_upload_preview` implementation**

In `ui_helpers.py`, replace the existing `show_upload_preview` function:

```python
def show_upload_preview(
    uploaded_file: object,
    total_pages: int = 0,
    selected: list[int] | None = None,
) -> None:
    """Show file info with page context as a single caption line.

    For >8 page PDFs: ``filename — size · Pages X–Y of Z``
    For ≤8 page PDFs: ``filename — size · N page(s) — All pages selected``
    """
    if uploaded_file is None:
        return

    name = getattr(uploaded_file, "name", "file")
    size = getattr(uploaded_file, "size", None)

    parts: list[str] = [f"**{name}**"]
    if size is not None:
        parts.append(f"{size / 1024:.0f} KB")
    info = " — ".join(parts)

    if total_pages > 0 and selected:
        page_start, page_end = selected[0], selected[-1]
        if total_pages > 8:
            info += f" · Pages {page_start}–{page_end} of {total_pages}"
        else:
            page_word = "page" if total_pages == 1 else "pages"
            info += f" · {total_pages} {page_word} — All pages selected"

    st.caption(info)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_helpers.py -v`

Expected: All `show_upload_preview` tests PASS. All other tests still PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check ui_helpers.py tests/test_ui_helpers.py
uv run ruff format ui_helpers.py tests/test_ui_helpers.py
git add ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: update show_upload_preview to include page info"
```

---

### Task 2: Restructure upload area and empty state

Replace the side-by-side upload + example button layout with a single-column upload area. Hide the question bar until a file is loaded. Remove the subtitle.

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Replace the upload layout**

In `streamlit_app.py`, replace lines 24–37 (title, subtitle, columns, uploader, example button):

```python
st.title("PDF Question & Answer")

uploaded_file = st.file_uploader(
    "Upload a PDF to get started",
    type=["pdf"],
    accept_multiple_files=False,
)
if st.button("Try with example", type="tertiary"):
    st.session_state["use_example_qa"] = True
    st.rerun()
```

This removes:
- `st.write("Upload a PDF, then ask a question.")`
- `col_upload, col_example = st.columns([3, 1])` and the `with` blocks
- The example button is now a de-emphasized `tertiary` button below the uploader

- [ ] **Step 2: Wrap the question bar and answer section in a file-loaded guard**

Currently the question `st.text_area` and `st.button("Answer")` are always visible. Wrap them so they only appear when `uploaded_file` is truthy.

Find the question text area (currently around line 151) and the answer button. The entire block from the question input through the answer display should be inside the existing `if uploaded_file:` block or a new one after file resolution. Structure:

```python
# After the file resolution block and page selection, add the guard:
if uploaded_file:
    col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_q:
        question = st.text_input(
            "Question",
            placeholder="Ask a question about this PDF...",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
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
                answer = generate_qa_response(
                    page_images, question, processor, model
                )

        if not answer:
            st.warning("Model produced no output.")
        else:
            st.session_state["last_answer"] = answer
            st.session_state["last_duration_s"] = t.duration_s
            st.session_state["last_source_pages"] = page_images
            st.session_state["last_page_numbers"] = selected
```

This replaces:
- `st.text_area("Question", ...)` → `st.text_input("Question", ..., label_visibility="collapsed")`
- `st.button("Answer", type="primary")` → `st.button("Ask", type="primary", use_container_width=True)`
- The "Upload a PDF first" warning is no longer needed since the button is hidden without a file

- [ ] **Step 3: Run the app to verify the empty state**

Run: `uv run streamlit run streamlit_app.py`

Verify:
- Page shows only title, uploader, and "Try with example" button when no file is loaded
- Question bar and Ask button appear only after uploading a PDF or clicking the example
- The question input and Ask button sit on one row

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: restructure upload area and hide question until file loaded"
```

---

### Task 3: Simplify page selection picker

Replace the `st.expander` with a thumbnail grid with a session-state-toggled container that shows only the range slider.

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Replace the page picker section**

Find the block starting at `if total_pages > 8:` (currently around line 76). Replace the entire expander block with:

```python
    if total_pages > 8:
        # Toggle picker via session state
        picker_key = "show_page_picker"
        if st.button(
            f"Pages {selected[0]}–{selected[-1]} of {total_pages} ✎",
            type="tertiary",
            key="page_range_toggle",
        ):
            st.session_state[picker_key] = not st.session_state.get(picker_key, False)
            st.rerun()

        if st.session_state.get(picker_key, False):
            with st.container(border=True):
                st.caption("Select up to 8 pages")
                file_size = getattr(uploaded_file, "size", 0)
                file_id = f"{getattr(uploaded_file, 'name', '')}_{total_pages}_{file_size}"
                if st.session_state.get("_slider_file_id") != file_id:
                    st.session_state.pop("page_range_slider", None)
                    st.session_state["_slider_file_id"] = file_id

                slider_range = st.select_slider(
                    "Select page range",
                    options=list(range(1, total_pages + 1)),
                    value=(1, min(total_pages, 8)),
                    key="page_range_slider",
                    label_visibility="collapsed",
                )
                if not isinstance(slider_range, tuple):
                    slider_range = (slider_range, slider_range)

                clamped = clamp_page_range(
                    slider_range[0], slider_range[1], max_span=8
                )
                if clamped != tuple(slider_range):
                    st.warning(
                        f"Maximum 8 pages — selection narrowed to pages {clamped[0]}–{clamped[1]}"
                    )
                    st.session_state["page_range_slider"] = clamped
                    st.rerun()

                selected = list(range(clamped[0], clamped[1] + 1))
                st.caption(
                    f"Pages {clamped[0]}–{clamped[1]} selected ({len(selected)} pages)"
                )
    else:
        st.caption(
            f"{total_pages} page{'s' if total_pages != 1 else ''} — All pages selected"
        )
```

This removes:
- `st.expander("Change pages")`
- All thumbnail caching logic (`file_key`, `st.session_state[file_key]`, batched rendering)
- `render_thumbnail_grid` call in the picker
- The `cols_per_row` calculation for the picker

- [ ] **Step 2: Remove unused thumbnail imports and session state cleanup**

In `streamlit_app.py`, remove `render_thumbnail_grid` from the import if it's no longer used anywhere (it will be re-added in Task 4 for the answer card). Also remove the old thumbnail cache cleanup in the file-change detection block (the `old_thumb_key` logic around lines 49-51) since thumbnails are no longer cached for the picker.

Remove from the file-change block:
```python
        # Clean up old thumbnail cache
        old_thumb_key = st.session_state.get("_thumb_key")
        if old_thumb_key:
            st.session_state.pop(old_thumb_key, None)
```

And remove `"_thumb_key"` references.

- [ ] **Step 3: Run the app and verify the page picker**

Run: `uv run streamlit run streamlit_app.py`

Verify with a >8 page PDF:
- File info line shows with a clickable page range button
- Clicking it toggles a container with just the range slider
- Clamping still works — selecting more than 8 pages shows a warning and narrows
- No thumbnail grid appears in the picker

Verify with a ≤8 page PDF:
- Shows caption like "3 pages — All pages selected", no toggle button

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: replace expander page picker with toggled slider-only container"
```

---

### Task 4: Redesign answer presentation

Move the answer into a styled card with a metadata footer. Collapse source page thumbnails behind a toggle.

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Replace the answer display section**

Find the "Display result" section (currently around line 193). Replace it with:

```python
# Display result
answer = st.session_state.get("last_answer")
if answer:
    with st.container(border=True):
        st.markdown(answer)

        duration = st.session_state.get("last_duration_s")
        source_pages = st.session_state.get("last_source_pages", [])
        page_numbers = st.session_state.get("last_page_numbers", [])

        # Footer row: metadata left, toggle right
        col_meta, col_toggle = st.columns([3, 1])
        with col_meta:
            meta_parts: list[str] = []
            if duration is not None:
                meta_parts.append(f"{duration:.1f}s")
            if page_numbers:
                meta_parts.append(f"Pages {page_numbers[0]}–{page_numbers[-1]}")
            if meta_parts:
                st.caption(" · ".join(meta_parts))

        with col_toggle:
            if source_pages:
                toggle_key = "show_source_pages"
                if st.button(
                    "Hide source pages" if st.session_state.get(toggle_key) else "Show source pages",
                    type="tertiary",
                    key="source_toggle",
                ):
                    st.session_state[toggle_key] = not st.session_state.get(
                        toggle_key, False
                    )
                    st.rerun()

        if source_pages and st.session_state.get("show_source_pages", False):
            render_thumbnail_grid(
                source_pages,
                selected_range=(1, len(source_pages)),
                cols_per_row=min(6, len(source_pages)),
            )
```

This replaces:
- The old `st.container(border=True)` + separate `st.caption` for duration
- The `st.divider()` + full-width page image grid
- Source pages are now inside the answer card, collapsed by default

- [ ] **Step 2: Ensure `render_thumbnail_grid` is imported**

In `streamlit_app.py`, confirm that `render_thumbnail_grid` is still in the imports from `ui_helpers`. It should be — it was imported originally and is now used in the answer card instead of the picker.

- [ ] **Step 3: Run the app and verify the answer card**

Run: `uv run streamlit run streamlit_app.py`

Upload a PDF, ask a question, verify:
- Answer appears in a bordered card
- Footer shows generation time and page range
- "Show source pages" button is visible
- Clicking it reveals thumbnails inside the card
- Clicking again hides them

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: redesign answer card with collapsed source pages"
```

---

### Task 5: Wire up `show_upload_preview` with new signature

Update the call site in `streamlit_app.py` to pass page context to the updated `show_upload_preview`.

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Update the `show_upload_preview` call**

Currently (around line 66):
```python
    show_upload_preview(uploaded_file)
```

Move this call to after `total_pages` and `selected` are computed, and pass the new arguments:

```python
    show_upload_preview(uploaded_file, total_pages=total_pages, selected=selected)
```

Note: the call must come after `total_pages` is computed (after `get_pdf_page_count`) and after `selected` is initialized (after `selected = list(range(1, auto_end + 1))`), but before the page picker toggle — so the info line appears above the picker.

- [ ] **Step 2: Run the app and verify the file info line**

Run: `uv run streamlit run streamlit_app.py`

Verify with a >8 page PDF: caption shows `**doc.pdf** — 245 KB · Pages 1–8 of 32`
Verify with a ≤8 page PDF: caption shows `**report.pdf** — 12 KB · 3 pages — All pages selected`

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -v`

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: wire show_upload_preview with page context"
```

---

### Task 6: Clean up stale state handling and final polish

Remove stale session state keys that are no longer used and ensure file changes clear the new state keys.

**Files:**
- Modify: `streamlit_app.py`

- [ ] **Step 1: Update the file-change detection block**

In the file-change detection block (where `new_file_id` is computed), update the keys that get cleared to include the new ones and remove the old thumbnail cache cleanup:

```python
    if st.session_state.get("_upload_file_id") != new_file_id:
        st.session_state["_upload_file_id"] = new_file_id
        for key in (
            "last_answer",
            "last_duration_s",
            "last_source_pages",
            "last_page_numbers",
            "show_page_picker",
            "show_source_pages",
            "page_range_slider",
            "_slider_file_id",
        ):
            st.session_state.pop(key, None)
```

- [ ] **Step 2: Run lint, format, and type check**

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
```

Fix any issues.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -v`

Expected: All tests PASS.

- [ ] **Step 4: Run the app end-to-end**

Run: `uv run streamlit run streamlit_app.py`

Full verification:
- Empty state: only title, uploader, and example button visible
- Upload a ≤8 page PDF: file info line shows all pages selected, no picker toggle, question bar appears
- Upload a >8 page PDF: file info line shows page range, clicking it opens slider-only picker, clamping works
- Ask a question: answer card appears with metadata footer, source pages collapsed
- Toggle source pages: thumbnails appear/disappear inside the card
- Switch files: all stale state is cleared (answer, picker, source toggle)
- Try with example: works as before

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "fix: clean up stale state keys for progressive disclosure UI"
```

---

### Task 7: Update CLAUDE.md

Update the project documentation to reflect the UI changes.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the UI section in CLAUDE.md**

In the `### UI` section and `### Key Details` section, update to reflect:
- Question bar uses `st.text_input` + "Ask" button in columns (not `st.text_area` + "Answer" button)
- Question bar is hidden until a file is uploaded (empty state)
- Page picker is a session-state-toggled container with slider only (not an expander with thumbnails)
- Thumbnail rendering/caching removed from page picker
- Answer displayed in a card with metadata footer and collapsible source pages
- Source page thumbnails shown via toggle inside the answer card (not always visible below a divider)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for progressive disclosure UI"
```
