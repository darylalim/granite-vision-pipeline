# Enterprise UI Improvements Design

## Context

The Granite Vision Pipeline is a Streamlit app for multipage document QA using granite-vision-3.3-2b. The target users are internal enterprise teams (analysts, researchers) on a shared deployment. The current UI is minimal: PDF upload, numeric page selectors, a text input, and plain markdown answer output.

This design addresses four pain points — usability, trust, productivity, and collaboration — via progressive enhancement of the existing single-page layout.

## Approach

Progressive enhancement on the existing single-page Streamlit app. No multi-page navigation. New features are added as self-contained sections within the top-to-bottom flow. Existing pipeline code (`pipeline/`) is unchanged; all changes are in the UI layer (`streamlit_app.py`, `ui_helpers.py`).

## Design

### 1. Page Layout & Upload Flow

Remove the help expander (`show_help`), sidebar model status (`show_sidebar_status`), and metrics bar (`show_metrics_bar`). Replace with inline cues. Retain `_ExampleFile`, `load_example`, and `show_upload_preview` — these are still used.

The page flows top-to-bottom in three phases:

1. **Upload section** — `st.file_uploader` (PDF only, single file) with "Try with example" button alongside. Below the uploader, an inline caption: "Upload a PDF (2+ pages)". After upload, show file preview via `show_upload_preview` (file name, size) plus page count displayed separately via `st.caption`.

2. **Page selection section** — Appears after a valid PDF (2+ pages) is uploaded. Thumbnail grid of all pages with a range slider for consecutive page selection.

3. **Question & Answer section** — Appears after pages are selected. Text area, Answer button with inline model status, tabbed answer/source display.

### 2. Thumbnail Grid Page Selection

Replace `st.number_input` start page / count controls with a visual thumbnail grid and range slider.

**Rendering:**
- After upload, render all pages as small thumbnails using `render_pdf_pages` at 72 DPI (half the current 144 DPI) for speed. This is a deliberate trade-off vs. the current selective-rendering approach: we render all pages upfront at low resolution so users can see what they're selecting. At 72 DPI, typical letter-size pages produce ~600x800px images, which is lightweight. For PDFs over 50 pages, render in batches of 20 to avoid blocking the UI.
- Cache thumbnails in `st.session_state` keyed by file identity so they don't re-render on every interaction.

**Layout:**
- Display thumbnails in a grid using `st.columns` — 4-6 per row depending on total page count.
- Each thumbnail captioned with its page number.

**Selection:**
- `st.select_slider` with a range below the thumbnail grid. Users drag two handles to select a consecutive range (e.g., pages 3-6).
- Selected thumbnails get a visible highlight using `st.container(border=True)` for selected pages (supported in Streamlit 1.29+). Avoid CSS injection where possible; use native Streamlit components.
- Caption below shows "Pages X-Y selected (N pages)".

**Validation:**
- After the user adjusts the slider, if the selected range spans more than 8 pages, automatically narrow it to the first 8 pages of the range, show `st.warning("Maximum 8 pages — selection narrowed to pages X-Y")`, and trigger `st.rerun()`.
- If fewer than 2 pages selected, disable the Answer button with `st.button("Answer", disabled=True)`.

### 3. Question & Answer Experience

**Question input:**
- Replace `st.text_input` with `st.text_area` for multi-sentence questions.
- Keep placeholder text. Add caption "Ctrl+Enter to submit".

**Model status:**
- Remove sidebar status display.
- On the Answer button, show spinners inline: "Loading model..." on first click (model load), "Generating answer..." on subsequent clicks.
- After first load, model is cached — only generation spinner appears.

**Answer display:**
- Show answer in `st.container(border=True)` for visual separation (native Streamlit, no CSS injection).
- Replace `show_metrics_bar` / `st.metric` duration display with a simple `st.caption("Generated in X.Xs")`.

**Conversation history:**
- Store Q&A pairs in `st.session_state` as a list of `{"question": str, "answer": str}` dicts.
- Display conversation history within the Answer tab (see Section 4), in chronological order, each as a Q/A block. The most recent answer appears at the bottom.
- Uploading a new PDF or changing page selection resets the history.

### 4. Trust — Source Page Verification

After answer generation, display results in `st.tabs(["Answer", "Source Pages"])`:

- **Answer tab** — generated answer text, duration caption, and conversation history.
- **Source Pages tab** — selected page images rendered at readable size (~400px height) in a horizontal layout using `st.columns`. Each image captioned with its page number.

Store the page images returned by `render_pdf_pages` (before QA resizing) in `st.session_state` for display in the Source Pages tab. These are the full-resolution rendered images, not the 768px-resized versions used for inference.

### 5. Productivity & Collaboration Foundations

**Download answer:**
- `st.download_button` below the answer tab.
- Exports Q&A as a markdown file via `format_qa_export(file_name: str, page_range: tuple[int, int], qa_pairs: list[dict[str, str]]) -> str`. Timestamp is generated internally as UTC ISO 8601. Example output:

```markdown
# QA Export — report.pdf (pages 3-6)
Generated: 2026-03-26T14:30:00Z

## Q&A

**Q:** What is the total revenue?
**A:** The total revenue is $1.2M as shown in the table on page 4.

**Q:** How does this compare to last quarter?
**A:** Revenue increased by 15% compared to Q3.
```

**Collaboration:**
- The download/export serves as the initial sharing mechanism (email/Slack).
- The markdown export format includes structured metadata so a future session history feature can ingest it.

**Explicitly out of scope:**
- User accounts or authentication
- Persistent database or cross-session storage
- Batch/bulk question processing
- Real-time collaboration or shared sessions

## Files Changed

- `streamlit_app.py` — restructure layout, add thumbnail grid, range slider, tabs, conversation history, download button, remove help expander and sidebar calls
- `ui_helpers.py` — remove `show_help`, `show_sidebar_status`, and `show_metrics_bar`; retain `_ExampleFile`, `load_example`, and `show_upload_preview`; add new helpers: `render_thumbnail_grid`, `clamp_page_range`, `format_qa_export`
- `tests/test_ui_helpers.py` — remove tests for `show_help`, `show_sidebar_status`, and `show_metrics_bar`; retain tests for `_ExampleFile`, `load_example`, and `show_upload_preview`; add tests for new helpers

## Files Unchanged

- `pipeline/` — no changes to models, PDF rendering, QA logic, or utilities
- `examples/` — no changes

## Testing Strategy

**New UI helpers** (tested with mocked `st`, same pattern as existing tests):
- `render_thumbnail_grid` — verify correct number of columns, image display calls, and caption text.
- `clamp_page_range(start: int, end: int, max_span: int) -> tuple[int, int]` — pure function, tested directly: ranges within limit pass through; ranges exceeding limit are narrowed to first `max_span` pages.
- `format_qa_export` — pure function returning a string. Test: correct markdown structure, file name, page range, Q&A pairs, UTC timestamp format.

**Behavioral logic extracted from `streamlit_app.py`:**
- Range slider validation and clamping logic is extracted into `clamp_page_range` in `ui_helpers.py` so it can be unit tested.
- Conversation history reset logic (triggered by file change or page selection change) is verified by testing session state manipulation in `test_ui_helpers.py`.

**Integration behavior in `streamlit_app.py`:**
- Layout flow, spinner states, and tab rendering are verified manually. These are thin Streamlit wiring, not business logic.

**Existing tests:**
- All pipeline tests (`test_models.py`, `test_utils.py`, `test_pdf.py`, `test_qa.py`) remain unchanged and must continue to pass.
- Retained UI helper tests (`_ExampleFile`, `load_example`, `show_upload_preview`) must continue to pass.
