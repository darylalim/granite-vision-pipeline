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
from ui_helpers import (
    load_example,
    show_help,
    show_metrics_bar,
    show_sidebar_status,
    show_upload_preview,
)

doctags_model = st.cache_resource(create_doctags_model)

EXAMPLE_IMAGE = "examples/sample.jpg"
EXAMPLE_PDF = "examples/sample.pdf"

st.title("Document Parsing")
st.write(
    "Parse document images to structured text in doctags format. "
    "Powered by IBM Granite Docling."
)

show_help(
    supported_formats="PNG, JPG, JPEG, PDF",
    description=(
        "Converts document images or PDF pages into doctags — a structured XML "
        "format that captures text, layout, and document structure. The doctags "
        "output can be further converted to Markdown. For PDFs, each page is "
        "processed independently."
    ),
    model_info="[granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)",
)

col_upload, col_ex_img, col_ex_pdf = st.columns([3, 1, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf"])
with col_ex_img:
    st.markdown("")  # spacing
    if st.button("Example image"):
        st.session_state["use_example_doctags"] = "image"
        st.rerun()
with col_ex_pdf:
    st.markdown("")  # spacing
    if st.button("Example PDF"):
        st.session_state["use_example_doctags"] = "pdf"
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_doctags", None)
elif st.session_state.get("use_example_doctags") == "image":
    active_file = load_example(EXAMPLE_IMAGE)
    st.caption("Using example image")
elif st.session_state.get("use_example_doctags") == "pdf":
    active_file = load_example(EXAMPLE_PDF)
    st.caption("Using example PDF")

if active_file:
    show_upload_preview(active_file)

is_pdf = active_file is not None and getattr(active_file, "name", "").lower().endswith(
    ".pdf"
)

if st.button("Generate", type="primary", disabled=not active_file):
    assert active_file is not None
    file_name = getattr(active_file, "name", "document")
    processor, model = doctags_model()
    st.session_state["model_docling"] = True

    if is_pdf:
        with temp_upload(active_file) as tmp_path:
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

            show_metrics_bar(
                {
                    "Pages": num_pages,
                    "Duration (s)": f"{t.duration_s:.2f}",
                }
            )

            combined_doctags = "\n\n".join(all_doctags)
            combined_markdown = "\n\n---\n\n".join(md for md in all_markdown if md)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                label="Download all doctags",
                data=combined_doctags,
                file_name=f"{file_name}_doctags.txt",
                mime="text/plain",
            )
            dl_col2.download_button(
                label="Download all Markdown",
                data=combined_markdown,
                file_name=f"{file_name}_doctags.md",
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
        image = Image.open(active_file).convert("RGB")

        with st.spinner("Generating doctags... This may take a few minutes."):
            with timed() as t:
                raw_doctags = generate_doctags(image, processor, model)

        show_metrics_bar({"Duration (s)": f"{t.duration_s:.2f}"})

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
                    file_name=f"{file_name}_doctags.md",
                    mime="text/markdown",
                )
            else:
                col_output.warning("Could not parse doctags into structured document.")

            col_output.download_button(
                label="Download raw doctags",
                data=raw_doctags,
                file_name=f"{file_name}_doctags.txt",
                mime="text/plain",
            )
        else:
            col_output.warning("Model produced no output.")

show_sidebar_status(
    models={"Docling": st.session_state.get("model_docling", False)},
)
