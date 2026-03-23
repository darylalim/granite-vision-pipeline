import streamlit as st

st.set_page_config(page_title="Granite Vision Pipeline")

pg = st.navigation(
    [
        st.Page("streamlit_home.py", title="Home", icon=":material/home:"),
        st.Page("pages/extraction.py", title="PDF Extraction"),
        st.Page("pages/segmentation.py", title="Image Segmentation"),
        st.Page("pages/doctags.py", title="Document Parsing"),
        st.Page("pages/qa.py", title="Multipage QA"),
        st.Page("pages/search.py", title="Document Search"),
    ]
)
pg.run()
