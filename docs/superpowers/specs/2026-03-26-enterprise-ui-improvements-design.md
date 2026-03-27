# Enterprise UI Improvements Design

## Context

The Granite Vision Pipeline is a Streamlit app for multipage document QA using granite-vision-3.3-2b. The target users are internal enterprise teams (analysts, researchers) on a shared deployment. The current UI is minimal: PDF upload, numeric page selectors, a text input, and plain markdown answer output.

This design addresses four pain points — usability, trust, productivity, and collaboration — via progressive enhancement of the existing single-page layout.

## Approach

Progressive enhancement on the existing single-page Streamlit app. No multi-page navigation. New features are added as self-contained sections within the top-to-bottom flow. Existing pipeline code (`pipeline/`) is unchanged; all changes are in the UI layer (`streamlit_app.py`, `ui_helpers.py`).

## Design

### 1. Page Layout & Upload Flow

Remove the help expander (`show_help`) and sidebar model status (`show_sidebar_status`). Replace with inline cues.

The page flows top-to-bottom in three phases:

1. **Upload section** — `st.file_uploader` (PDF only, single file) with "Try with example" button alongside. Below the uploader, an inline caption: "Upload a PDF (2+ pages)". After upload, show file name, size, and page count.

2. **Page selection section** — Appears after a valid PDF (2+ pages) is uploaded. Thumbnail grid of all pages with a range slider for consecutive page selection.

3. **Question & Answer section** — Appears after pages are selected. Text area, Answer button with inline model status, tabbed answer/source display.

### 2. Thumbnail Grid Page Selection

Replace `st.number_input` start page / count controls with a visual thumbnail grid and range slider.

**Rendering:**
- After upload, render all pages as small thumbnails using `render_pdf_pages` at 72 DPI (half the current 144 DPI) for speed.
- Cache thumbnails in `st.session_state` keyed by file identity so they don't re-render on every interaction.

**Layout:**
- Display thumbnails in a grid using `st.columns` — 4-6 per row depending on total page count.
- Each thumbnail captioned with its page number.

**Selection:**
- `st.select_slider` with a range below the thumbnail grid. Users drag two handles to select a consecutive range (e.g., pages 3-6).
- Selected thumbnails get a visible highlight via CSS injection (`st.markdown` with `unsafe_allow_html=True`) — e.g., a colored border.
- Caption below shows "Pages X-Y selected (N pages)".

**Validation:**
- Slider range clamped to max 8 pages. If the range exceeds 8, show `st.warning` and clamp.
- If fewer than 2 pages selected, disable the Answer button.

### 3. Question & Answer Experience

**Question input:**
- Replace `st.text_input` with `st.text_area` for multi-sentence questions.
- Keep placeholder text. Add caption "Ctrl+Enter to submit".

**Model status:**
- Remove sidebar status display.
- On the Answer button, show spinners inline: "Loading model..." on first click (model load), "Generating answer..." on subsequent clicks.
- After first load, model is cached — only generation spinner appears.

**Answer display:**
- Show answer in a styled `st.container` with light background via CSS injection.
- Replace `st.metric` duration display with a simple caption: "Generated in X.Xs".

**Conversation history:**
- Store Q&A pairs in `st.session_state` as a list of `{"question": str, "answer": str}` dicts.
- Display previous exchanges above the current answer in chronological order, each as a Q/A block.
- Uploading a new PDF or changing page selection resets the history.

### 4. Trust — Source Page Verification

After answer generation, display results in `st.tabs(["Answer", "Source Pages"])`:

- **Answer tab** — generated answer text, duration caption, and conversation history.
- **Source Pages tab** — selected page images rendered at readable size (~400px height) in a horizontal layout using `st.columns`. Each image captioned with its page number.

Source page images are already rendered during QA. Store them in `st.session_state` rather than discarding after inference.

### 5. Productivity & Collaboration Foundations

**Download answer:**
- `st.download_button` below the answer tab.
- Exports Q&A as a markdown file containing: source file name, selected page range, all Q&A pairs from the session, and a timestamp.

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
- `ui_helpers.py` — remove `show_help` and `show_sidebar_status`; add new helpers: `render_thumbnail_grid`, `highlight_selected_css`, `format_qa_export`
- `tests/test_ui_helpers.py` — remove tests for `show_help` and `show_sidebar_status`; add tests for new helpers

## Files Unchanged

- `pipeline/` — no changes to models, PDF rendering, QA logic, or utilities
- `examples/` — no changes

## Testing Strategy

- New UI helpers (`render_thumbnail_grid`, `highlight_selected_css`, `format_qa_export`) are pure functions or thin Streamlit wrappers — tested with mocked `st` the same way existing helpers are tested.
- Thumbnail caching logic tested by verifying `st.session_state` is read/written correctly.
- Conversation history tested by simulating multiple Q&A cycles in session state.
- Existing pipeline tests remain unchanged and must continue to pass.
