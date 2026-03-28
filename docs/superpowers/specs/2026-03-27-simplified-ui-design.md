# Simplified UI Design

## Goal

Simplify the Granite Vision Pipeline UI for non-technical end users. Strip to the core experience: upload a PDF, ask a question, get an answer. Remove features that add cognitive overhead without clear value for this audience.

## Audience

Non-technical end users who need to ask questions about PDF documents.

## Core Flow

Three actions: **Upload -> Ask -> Answer**.

1. User uploads a PDF (or clicks "Try with example").
2. User types a question.
3. User clicks "Answer" and sees the result.

No other interaction is required for the common case.

## Layout (top to bottom)

```
Title: "PDF Question & Answer"
Subtitle: "Upload a PDF, then ask a question."

[File uploader]  [Try with example]
caption: "document.pdf -- 12 pages -- Pages 1-8 selected"

> Change pages          (expander, only if >8 pages)
  [Range slider]
  [Thumbnail grid with selection highlighting]

[Question text area]
[Answer button]

Answer text in a bordered container
caption: "Generated in 1.23s"

Source page images inline, labeled "Page 1", "Page 2", ...
```

## Page Selection

- On upload, auto-select the first N pages where N = min(total_pages, 8).
- For PDFs with 8 or fewer pages, all pages are selected and no page picker is shown.
- For PDFs with more than 8 pages, a collapsed "Change pages" expander appears below the upload caption. Inside: the range slider (clamped to max 8 pages) and a thumbnail grid with selection highlighting.
- 1-page PDFs are supported (no minimum page requirement in the UI).

## Answer Display

- Single answer displayed inline (not in tabs). Each new answer replaces the previous one.
- Answer text in a bordered container with a "Generated in X.XXs" caption.
- Source page images shown inline below the answer, labeled with page numbers.

## Pipeline Change

- `generate_qa_response()` in `pipeline/qa.py` updated to accept 1-8 images (currently requires 2-8).
- Validation error message updated accordingly.

## Removed Features

- **Conversation history** -- no `qa_history` session state, no multi-turn display.
- **Tabbed display** (Answer / Source Pages) -- replaced by inline sequential layout.
- **Q&A export / download button** -- removed entirely.
- **`format_qa_export()`** in `ui_helpers.py` -- removed (no export feature).
- **Thumbnail grid on the main view** -- only inside the "Change pages" expander for >8 page PDFs.

## Kept Features

- Model caching via `st.cache_resource`.
- Temporary file handling (`temp_upload`).
- Image resizing for GPU memory (`resize_for_qa`).
- PDF rendering and page counting.
- "Try with example" button.
- Post-click validation (missing PDF, missing question).

## Files Affected

- `streamlit_app.py` -- major rewrite to simplified layout.
- `ui_helpers.py` -- remove `format_qa_export()`.
- `pipeline/qa.py` -- change validation from 2-8 to 1-8 images.
- `tests/test_qa.py` -- update validation tests for 1-image minimum.
- `tests/test_ui_helpers.py` -- remove `format_qa_export` tests.
- `CLAUDE.md` -- update architecture documentation to reflect changes.
