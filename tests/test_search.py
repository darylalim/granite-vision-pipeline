"""Tests for the search module."""

from pathlib import Path
from unittest.mock import MagicMock, patch


# --- create_embedding_model tests ---


@patch("pipeline.search.SentenceTransformer")
def test_create_embedding_model_loads_correct_model(
    mock_st_cls: MagicMock,
) -> None:
    from pipeline.search import create_embedding_model

    model = create_embedding_model()

    mock_st_cls.assert_called_once_with("ibm-granite/granite-embedding-english-r2")
    assert model is mock_st_cls.return_value


# --- get_collection tests ---


def test_get_collection_returns_collection() -> None:
    from pipeline.search import COLLECTION_NAME, get_collection

    collection = get_collection()
    assert collection.name == COLLECTION_NAME


def test_get_collection_uses_persistent_client(tmp_path: Path) -> None:
    from pipeline.search import get_collection

    with patch("pipeline.search.CHROMA_PATH", str(tmp_path / "test_chroma")):
        collection = get_collection()
        assert (tmp_path / "test_chroma").exists()
        assert collection is not None
