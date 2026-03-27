# Granite Vision Pipeline

Streamlit web app for multipage document question answering. Upload a PDF, select 2-8 consecutive pages, ask a question, and get a text answer.

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Requires Python 3.12+ and Streamlit 1.55.0+. Includes a "Try with example" button for a quick demo.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## How It Works

Upload a PDF and select 2-8 consecutive pages using the thumbnail grid and range slider. Type a question and the model will analyze all selected pages together to generate a text answer. Page images are resized to 768px max dimension for GPU memory efficiency.

Results are displayed in a tabbed view: the Answer tab shows conversation history across multiple questions, and the Source Pages tab shows the exact pages the model processed. Q&A sessions can be exported as markdown.

## Project Structure

```
streamlit_app.py         # single-page Streamlit app
ui_helpers.py            # shared UI functions (preview, thumbnails, export, examples)
pipeline/
  __init__.py            # public API re-exports
  models.py              # model loading, vision model factory, generate helper
  utils.py               # temp_upload and timed context managers
  pdf.py                 # PDF rendering and page count utilities
  qa.py                  # image resizing, multipage QA inference (2-8 pages)
examples/
  sample.pdf             # sample PDF for demo mode
tests/
  test_models.py         # model loading and generate_response tests
  test_utils.py          # temp_upload and timed context manager tests
  test_pdf.py            # PDF rendering and page count tests
  test_qa.py             # QA resizing, prompt structure, and validation tests
  test_ui_helpers.py     # UI helpers, thumbnails, export, and example file tests
```
