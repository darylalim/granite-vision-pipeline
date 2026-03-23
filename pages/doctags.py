import streamlit as st
from docling_core.types.doc.document import DoclingDocument
from PIL import Image

from pipeline import (
    create_doctags_model,
    export_markdown,
    generate_doctags,
    parse_doctags,
    render_pdf_pages,
    temp_upload,
    timed,
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

    if is_pdf:
        with temp_upload(uploaded_file) as tmp_path:
            with st.spinner("Rendering PDF pages..."):
                page_images = render_pdf_pages(tmp_path)

            num_pages = len(page_images)
            progress = st.progress(0, text="Generating doctags...")

            all_doctags: list[str] = []
            all_markdown: list[str] = []
            all_docs: list[DoclingDocument | None] = []

            with timed() as t:
                for i, page_image in enumerate(page_images):
                    progress.progress(
                        (i + 1) / num_pages,
                        text=f"Processing page {i + 1} of {num_pages}...",
                    )
                    raw = generate_doctags(page_image, processor, model)
                    all_doctags.append(raw)

                    doc = parse_doctags(raw, page_image) if raw else None
                    all_docs.append(doc)
                    all_markdown.append(export_markdown(doc) if doc else "")

            progress.empty()

            col1, col2 = st.columns(2)
            col1.metric("Pages", num_pages)
            col2.metric("Duration (s)", f"{t.duration_s:.2f}")

            combined_doctags = "\n\n".join(all_doctags)
            combined_markdown = "\n\n---\n\n".join(md for md in all_markdown if md)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                label="Download all doctags",
                data=combined_doctags,
                file_name=f"{uploaded_file.name}_doctags.txt",
                mime="text/plain",
            )
            dl_col2.download_button(
                label="Download all Markdown",
                data=combined_markdown,
                file_name=f"{uploaded_file.name}_doctags.md",
                mime="text/markdown",
            )

            for i, page_image in enumerate(page_images):
                with st.expander(f"Page {i + 1}", expanded=i == 0):
                    col_img, col_output = st.columns(2)
                    col_img.image(page_image, caption=f"Page {i + 1}")

                    if all_doctags[i]:
                        col_output.code(all_doctags[i], language="xml")

                        if all_docs[i] is not None:
                            col_output.markdown("**Markdown output:**")
                            col_output.markdown(all_markdown[i])
                        else:
                            col_output.warning(
                                "Could not parse doctags into structured document."
                            )
                    else:
                        col_output.warning("Model produced no output for this page.")

    else:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            with timed() as t:
                raw_doctags = generate_doctags(image, processor, model)

        st.metric("Duration (s)", f"{t.duration_s:.2f}")

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
