"""Shared UI helper functions for Streamlit pages.

No pipeline imports — any data requiring pipeline calls is computed
at the call site and passed as a parameter.
"""

import io
from pathlib import Path

import streamlit as st


def show_upload_preview(uploaded_file: object) -> None:
    """Show preview of an uploaded PDF file (filename and size)."""
    if uploaded_file is None:
        return

    name = getattr(uploaded_file, "name", "file")
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        st.caption(f"**{name}**\n{size / 1024:.0f} KB")
    else:
        st.caption(f"**{name}**")


def show_help(
    supported_formats: str,
    description: str,
    model_info: str,
) -> None:
    """Render a collapsed 'How this works' expander with consistent formatting."""
    with st.expander("How this works"):
        st.markdown(f"**Supported formats:** {supported_formats}")
        st.markdown(description)
        st.markdown(f"**Model:** {model_info}")


def show_metrics_bar(metrics: dict[str, object]) -> None:
    """Render metrics in equal columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


class _ExampleFile(io.BytesIO):
    """BytesIO wrapper with .name and .size attributes for UploadedFile compat."""

    def __init__(self, data: bytes, name: str = "", size: int = 0) -> None:
        super().__init__(data)
        self.name = name
        self.size = size


def load_example(file_path: str) -> _ExampleFile:
    """Load a file as a BytesIO wrapper that substitutes for st.UploadedFile."""
    data = Path(file_path).read_bytes()
    return _ExampleFile(data, name=Path(file_path).name, size=len(data))


def show_sidebar_status(
    models: dict[str, bool],
) -> None:
    """Show model status in the sidebar."""
    with st.sidebar:
        st.markdown("**Models**")
        for name, loaded in models.items():
            status = "Loaded" if loaded else "Not loaded"
            st.text(f"{name}: {status}")
