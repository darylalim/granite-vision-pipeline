import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from docling.exceptions import ConversionError

from pipeline import build_output, convert, create_converter, get_description

converter = st.cache_resource(create_converter)

st.title("Granite Vision Pipeline")
st.write("Describe pictures in PDF documents using IBM Granite Vision.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])

if st.button("Annotate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    tmp_path: str | None = None
    try:
        with st.spinner(
            "Extracting content... This may take a few minutes for large documents."
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            start = time.perf_counter_ns()
            doc = convert(tmp_path, converter=converter())
            duration_s = (time.perf_counter_ns() - start) / 1e9

        st.success("Done.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pictures", len(doc.pictures))
        col2.metric("Tables", len(doc.tables))
        col3.metric("Duration (s)", f"{duration_s:.2f}")

        st.download_button(
            label="Download JSON",
            data=json.dumps(build_output(doc, duration_s), indent=2),
            file_name=f"{uploaded_file.name}_annotations.json",
            mime="application/json",
        )

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

    except ConversionError as e:
        st.error(str(e))
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
