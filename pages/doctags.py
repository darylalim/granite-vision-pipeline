import time

import streamlit as st
from PIL import Image

from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
)

doctags_model = st.cache_resource(create_doctags_model)

st.title("DocTags Generation (Experimental)")
st.write(
    "Parse document images to structured text in doctags format. "
    "Powered by IBM Granite Docling."
)

uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf"])

is_pdf = uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf")

if st.button("Generate", type="primary", disabled=not uploaded_file):
    assert uploaded_file is not None
    processor, model = doctags_model()

    if not is_pdf:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            start = time.perf_counter_ns()
            raw_doctags = generate_doctags(image, processor, model)
            duration_s = (time.perf_counter_ns() - start) / 1e9

        st.metric("Duration (s)", f"{duration_s:.2f}")

        col_img, col_output = st.columns(2)
        col_img.image(image, caption="Original")

        if raw_doctags:
            col_output.code(raw_doctags, language="xml")

            doc = parse_doctags(raw_doctags, image)
            if doc is not None:
                md = export_markdown(doc)
                col_output.markdown("**Markdown output:**")
                col_output.markdown(md)
                col_output.download_button(
                    label="Download Markdown",
                    data=md,
                    file_name="doctags_output.md",
                    mime="text/markdown",
                )
            else:
                col_output.warning("Could not parse doctags into structured document.")

            col_output.download_button(
                label="Download raw doctags",
                data=raw_doctags,
                file_name="doctags_output.txt",
                mime="text/plain",
            )
        else:
            col_output.warning("Model produced no output.")
