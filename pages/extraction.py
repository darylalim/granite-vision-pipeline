import json

import streamlit as st
from docling.exceptions import ConversionError
from docling_core.types.doc.document import DoclingDocument

from pipeline import (
    build_output,
    convert,
    create_converter,
    get_description,
    temp_upload,
    timed,
)
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

converter = st.cache_resource(create_converter)

EXAMPLE_PDF = "examples/sample.pdf"

st.title("PDF Extraction")
st.write(
    "Extract and describe pictures and tables in PDF documents using IBM Granite Vision."
)

show_help(
    supported_formats="PDF",
    description=(
        "Uploads a PDF and uses Docling to extract pictures and tables. "
        "Pictures are described using Granite Vision. Tables are parsed into "
        "structured data. Results are available as a JSON download."
    ),
    model_info="[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) via Docling",
)

col_upload, col_example = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader("Upload file", type=["pdf"])
with col_example:
    if st.button("Try with example"):
        st.session_state["use_example_extraction"] = True
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_extraction", None)
elif st.session_state.get("use_example_extraction"):
    active_file = load_example(EXAMPLE_PDF)
    st.caption("Using example file")

if active_file:
    show_upload_preview(active_file)


def _render_elements(doc: DoclingDocument) -> None:
    """Display extracted pictures and tables in expanders."""
    for idx, pic in enumerate(doc.pictures, 1):
        with st.expander(f"Picture {idx}", expanded=idx == 1):
            col_img, col_desc = st.columns(2)
            image = pic.get_image(doc)
            if image:
                col_img.image(image)
            caption = pic.caption_text(doc=doc)
            if caption:
                col_img.caption(caption)
            desc = get_description(pic)
            if desc:
                col_desc.markdown(desc["text"])
            else:
                col_desc.write("No description available.")

    for idx, table in enumerate(doc.tables, 1):
        with st.expander(
            f"Table {idx}",
            expanded=len(doc.pictures) == 0 and idx == 1,
        ):
            col_img, col_data = st.columns(2)
            image = table.get_image(doc)
            if image:
                col_img.image(image)
            caption = table.caption_text(doc=doc)
            if caption:
                col_img.caption(caption)
            df = table.export_to_dataframe(doc=doc)
            if not df.empty:
                col_data.dataframe(df)
            else:
                col_data.write("Empty table.")


if st.button("Annotate", type="primary", disabled=not active_file):
    assert active_file is not None
    file_name = getattr(active_file, "name", "document.pdf")
    try:
        with temp_upload(active_file) as tmp_path:
            with st.spinner(
                "Extracting content... This may take a few minutes for large documents."
            ):
                with timed() as t:
                    doc = convert(tmp_path, converter=converter())
                    st.session_state["model_docling"] = True

        st.success("Done.")

        show_metrics_bar(
            {
                "Pictures": len(doc.pictures),
                "Tables": len(doc.tables),
                "Duration (s)": f"{t.duration_s:.2f}",
            }
        )

        output = build_output(doc, t.duration_s)

        st.download_button(
            label="Download JSON",
            data=json.dumps(output, indent=2),
            file_name=f"{file_name}_annotations.json",
            mime="application/json",
        )

        _render_elements(doc)

    except ConversionError as e:
        st.error(str(e))

show_sidebar_status(
    models={"Docling": st.session_state.get("model_docling", False)},
)
