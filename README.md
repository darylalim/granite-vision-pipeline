# Granite Vision Pipeline

Streamlit web app with two capabilities:

1. **PDF Extraction** — extract and describe pictures and tables in PDF documents
2. **Multipage QA** — answer questions across up to 8 document pages

Powered by [granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b). Navigate between features using the sidebar. Each page includes a "Try with example" button for quick demos.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Features

**PDF Extraction** — Upload a PDF to extract pictures with AI-generated descriptions and tables with structured data. Results available as JSON download with per-element previews.

**Multipage QA** — Upload a PDF or up to 8 images and ask questions about the content. Images are resized to 768px max dimension for GPU memory efficiency. Answers are displayed alongside page thumbnails.

## Project Structure

```
streamlit_app.py         # navigation hub (st.navigation routing)
ui_helpers.py            # shared UI functions (preview, help, metrics, examples, sidebar)
pipeline/
  __init__.py            # public API re-exports
  models.py              # shared model loading, vision model factory, generate helper
  utils.py               # temp_upload and timed context managers
  config.py              # converter factory, convert wrapper
  output.py              # unified element builder, description and table extraction
  pdf.py                 # PDF rendering and page count utilities
  qa.py                  # image resizing, multipage QA inference
pages/
  extraction.py          # PDF extraction page
  qa.py                  # multipage QA page
examples/
  sample.pdf             # sample PDF for demo mode
tests/
  test_config.py         # converter factory and pipeline option tests
  test_output.py         # element builder, description, and table content tests
  test_models.py         # shared model loading and generate_response tests
  test_utils.py          # temp_upload and timed context manager tests
  test_pdf.py            # PDF rendering and page count tests
  test_qa.py             # QA resizing, prompt structure, and validation tests
  test_ui_helpers.py     # UI helper functions and example file loading tests
```
