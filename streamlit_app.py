import streamlit as st

st.set_page_config(page_title="Granite Vision Pipeline")

pg = st.navigation(
    [
        st.Page("pages/extraction.py", title="PDF Extraction", icon=":material/home:"),
        st.Page("pages/qa.py", title="Multipage QA"),
    ]
)
pg.run()
