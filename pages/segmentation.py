import io

import streamlit as st
from PIL import Image

from pipeline import create_granite_vision_model, create_sam_model, draw_mask, segment

granite_model = st.cache_resource(create_granite_vision_model)
sam_model = st.cache_resource(create_sam_model)

st.title("Image Segmentation (Experimental)")
st.write(
    "Segment objects in images using natural language prompts. "
    "Powered by Granite Vision with SAM refinement."
)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Segmentation prompt", placeholder="e.g., the dog on the left")

if st.button("Segment", type="primary", disabled=not uploaded_file or not prompt):
    assert uploaded_file is not None
    image = Image.open(uploaded_file)

    with st.spinner("Running segmentation... This may take a few minutes."):
        mask = segment(
            image,
            prompt,
            granite=granite_model(),
            sam=sam_model(),
        )

    if mask is None:
        st.error("Segmentation failed — no mask found in model output.")
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
            file_name="segmentation_mask.png",
            mime="image/png",
        )
