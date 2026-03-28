# Progressive Disclosure UI Redesign

**Date:** 2026-03-28
**Goal:** Simplify the UI for non-technical users (business analysts, students) who primarily do quick one-off document questions but may ask follow-ups.

## Design Principles

- One clear action at a time — hide controls until they're relevant
- Progressive disclosure — simple by default, powerful when needed
- Minimal cognitive load — reduce visible elements at every stage

## 1. Layout & Flow

Replace the current side-by-side upload + example button layout with a vertically stacked flow:

- **Title only** — remove the subtitle ("Upload a PDF, then ask a question")
- **Upload area** — larger, centered drop zone with "Try with an example" as an inline link beneath "or", instead of a separate button in a second column
- **File info line** — after upload, show `filename — size · Pages X–Y of Z ✎` on a single line. The page range is a clickable link that opens the picker (for >8 page PDFs). For ≤8 page PDFs, show plain text like `3 pages — All pages selected` with no link
- **Question bar** — text input and "Ask" button side-by-side on one row (search bar pattern), replacing the separate text area + "Answer" button below it
- **Answer card** — appears below the question bar after submission

## 2. Empty State

Hide the question input and Ask button until a PDF is uploaded. Before upload, the page shows only:

- Title
- Upload drop zone with integrated example link

This eliminates the dead-end UI where users see a question field and Answer button that do nothing without a file.

## 3. Page Selection

Only shown for PDFs with more than 8 pages. Triggered by clicking the page range link in the file info line.

- **Default:** page range shown as clickable text in the file info line (`Pages 1–8 of 32 ✎`)
- **Expanded:** opens a container with the range slider only. No thumbnail grid in the picker — thumbnails are removed from the selection step entirely
- **Summary:** single line below slider: `Pages X–Y selected (N pages)`
- **Clamping:** same max-8-page clamping logic, with a warning if the selection is narrowed

Changes from current:
- Remove the `st.expander("Change pages")` — replace with a container toggled via `st.session_state` flag, shown/hidden by a `st.button` styled as the page range link
- Remove thumbnail rendering and caching in the picker (`render_thumbnail_grid` no longer called during page selection)
- Remove batched thumbnail rendering for >50 page PDFs in the picker

## 4. Answer Presentation

- **Answer card** — bordered/styled card containing the answer text with a footer row inside the card
- **Footer row** — generation time and page range on the left (`2.3s · Pages 1–8`), "Show source pages" toggle on the right
- **Source pages collapsed by default** — clicking the toggle reveals compact thumbnails inside the card (reusing `render_thumbnail_grid`)
- **Compact thumbnails** — smaller than current, shown in a wrapped row inside the answer card rather than full-width columns below a divider
- Each new question replaces the previous answer (same as current)

Changes from current:
- Move source page images from always-visible grid below a divider into a collapsible section inside the answer card
- Replace `st.caption` for generation time with inline text in the card footer
- Remove the `st.divider()` between answer and source pages

## 5. Component Changes Summary

### `streamlit_app.py`
- Restructure layout: upload area centered, question bar hidden until file loaded
- Replace `col_upload, col_example = st.columns([3, 1])` with single-column layout
- Integrate "Try with example" as a small `st.button` below the uploader (Streamlit doesn't support inline links that trigger actions, so a button is used but visually de-emphasized)
- Replace `st.text_area` + separate `st.button("Answer")` with inline `st.text_input` + `st.button("Ask")` in columns
- Replace `st.expander("Change pages")` with toggled container via session state
- Remove thumbnail rendering from page picker section
- Move source page display into answer card with collapsible toggle
- Hide question input when no file is uploaded

### `ui_helpers.py`
- `render_thumbnail_grid` — still used for source page display in the answer card, but no longer called in the page picker
- `show_upload_preview` — update to produce the merged file info + page range line instead of just filename/size
- `clamp_page_range` — no changes

### Removed from page picker flow
- Thumbnail caching in `st.session_state` for the picker
- Batched thumbnail rendering for >50 page PDFs in the picker
- `render_thumbnail_grid` call inside the page expander

## 6. Testing Impact

- `test_ui_helpers.py` — update `show_upload_preview` tests for new output format. `render_thumbnail_grid` tests remain (still used for source pages). `clamp_page_range` tests unchanged.
- No pipeline test changes — `pipeline/` module is unaffected.
