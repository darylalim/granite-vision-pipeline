import io

import streamlit as st
from PIL import Image

from pipeline import create_granite_vision_model, create_sam_model, draw_mask, segment
from ui_helpers import (
    load_example,
    show_help,
    show_sidebar_status,
    show_upload_preview,
)

granite_model = st.cache_resource(create_granite_vision_model)
sam_model = st.cache_resource(create_sam_model)

EXAMPLE_IMAGE = "examples/sample.jpg"

st.title("Image Segmentation")
st.write(
    "Segment objects in images using natural language prompts. "
    "Powered by Granite Vision with SAM refinement."
)

show_help(
    supported_formats="PNG, JPG, JPEG",
    description=(
        "Upload an image and describe the object you want to segment. "
        "The model identifies the object and produces a binary mask. "
        "For best results, use specific descriptions like "
        "'the red car on the left' instead of just 'car'."
    ),
    model_info=(
        "[granite-vision-3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) "
        "+ [SAM (sam-vit-huge)](https://huggingface.co/facebook/sam-vit-huge)"
    ),
)

col_upload, col_example = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
with col_example:
    if st.button("Try with example"):
        st.session_state["use_example_segmentation"] = True
        st.rerun()

# Resolve file: user upload takes priority over example
active_file = uploaded_file
if uploaded_file:
    st.session_state.pop("use_example_segmentation", None)
elif st.session_state.get("use_example_segmentation"):
    active_file = load_example(EXAMPLE_IMAGE)
    st.caption("Using example file")

if active_file:
    show_upload_preview(active_file)

default_prompt = (
    "the red rectangle" if st.session_state.get("use_example_segmentation") else ""
)
prompt = st.text_input(
    "Segmentation prompt",
    value=default_prompt,
    placeholder="e.g., the dog on the left",
)

if st.button("Segment", type="primary", disabled=not active_file or not prompt):
    assert active_file is not None
    file_name = getattr(active_file, "name", "image")
    image = Image.open(active_file)

    with st.spinner("Running segmentation... This may take a few minutes."):
        mask = segment(
            image,
            prompt,
            granite=granite_model(),
            sam=sam_model(),
        )
        st.session_state["model_granite_vision"] = True
        st.session_state["model_sam"] = True

    if mask is None:
        st.error(
            "Couldn't find that object. Try a more specific description, "
            "like 'the red car on the left' instead of 'car'."
        )
    else:
        col_orig, col_overlay = st.columns(2)
        col_orig.image(image, caption="Original")
        overlay = draw_mask(mask, image)
        col_overlay.image(overlay, caption="Segmentation overlay")

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        st.download_button(
            label="Download mask",
            data=buf.getvalue(),
            file_name=f"{file_name}_mask.png",
            mime="image/png",
        )

show_sidebar_status(
    models={
        "Granite Vision": st.session_state.get("model_granite_vision", False),
        "SAM": st.session_state.get("model_sam", False),
    },
)
