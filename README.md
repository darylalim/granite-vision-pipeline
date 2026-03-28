# Granite Vision Pipeline

Streamlit web app for document question answering. Upload a PDF, ask a question, and get a text answer.

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Requires Python 3.12+ and Streamlit 1.55.0+. Includes a "Try with example" button for a quick demo.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## How It Works

Upload a PDF and ask a question. The app automatically selects up to 8 pages for the model to analyze. For PDFs with more than 8 pages, an optional page range picker lets you choose which pages to use. Page images are resized to 768px max dimension for GPU memory efficiency.

The answer is displayed inline with the source pages shown below for verification.

## Project Structure

```
streamlit_app.py         # single-page Streamlit app
ui_helpers.py            # shared UI functions (preview, thumbnails, examples)
pipeline/
  __init__.py            # public API re-exports
  models.py              # model loading, vision model factory, generate helper
  utils.py               # temp_upload and timed context managers
  pdf.py                 # PDF rendering and page count utilities
  qa.py                  # image resizing, QA inference (1-8 pages)
examples/
  sample.pdf             # sample PDF for demo mode
tests/
  test_models.py         # model loading and generate_response tests
  test_utils.py          # temp_upload and timed context manager tests
  test_pdf.py            # PDF rendering and page count tests
  test_qa.py             # QA resizing, prompt structure, and validation tests
  test_ui_helpers.py     # UI helpers, thumbnails, and example file tests
```
