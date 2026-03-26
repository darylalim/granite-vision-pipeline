# Granite Vision Pipeline

Streamlit web app for multipage document question answering. Upload a PDF or up to 8 images, ask a question, and get an answer across all pages.

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Includes a "Try with example" button for a quick demo.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## How It Works

Upload a single PDF or up to 8 images and ask a question about the content. For PDFs, select which pages to include (up to 8). Images are resized to 768px max dimension for GPU memory efficiency. Answers are displayed alongside page thumbnails.

## Project Structure

```
streamlit_app.py         # single-page Streamlit app
ui_helpers.py            # shared UI functions (preview, help, metrics, examples, sidebar)
pipeline/
  __init__.py            # public API re-exports
  models.py              # model loading, vision model factory, generate helper
  utils.py               # temp_upload and timed context managers
  pdf.py                 # PDF rendering and page count utilities
  qa.py                  # image resizing, multipage QA inference
examples/
  sample.pdf             # sample PDF for demo mode
tests/
  test_models.py         # model loading and generate_response tests
  test_utils.py          # temp_upload and timed context manager tests
  test_pdf.py            # PDF rendering and page count tests
  test_qa.py             # QA resizing, prompt structure, and validation tests
  test_ui_helpers.py     # UI helper functions and example file loading tests
```
