"""Tests for the search module."""

import chromadb
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


# --- clear_collection tests ---


def test_clear_collection_empties_collection() -> None:
    from pipeline.search import clear_collection

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_clear")
    collection.upsert(ids=["1", "2"], documents=["doc1", "doc2"])
    assert collection.count() == 2

    clear_collection(collection)
    assert collection.count() == 0


# --- _extract_text helper tests ---


def test_extract_text_from_picture_description() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "",
        "content": {"description": {"text": "A bar chart.", "created_by": "model"}},
    }
    assert _extract_text(element) == "A bar chart."


def test_extract_text_from_table_markdown() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "table",
        "caption": "",
        "content": {
            "markdown": "| Col |\n|---|\n| val |",
            "data": {"columns": ["Col"], "rows": [["val"]]},
        },
    }
    assert _extract_text(element) == "| Col |\n|---|\n| val |"


def test_extract_text_prepends_caption() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "Figure 1: Revenue",
        "content": {"description": {"text": "A chart.", "created_by": "model"}},
    }
    assert _extract_text(element) == "Figure 1: Revenue\nA chart."


def test_extract_text_returns_none_for_no_description() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "picture",
        "caption": "",
        "content": {"description": None},
    }
    assert _extract_text(element) is None


def test_extract_text_returns_none_for_empty_markdown() -> None:
    from pipeline.search import _extract_text

    element = {
        "type": "table",
        "caption": "",
        "content": {"markdown": "", "data": {"columns": [], "rows": []}},
    }
    assert _extract_text(element) is None


# --- index_elements tests ---


def test_index_elements_picture() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_index_pic")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "A chart.", "created_by": "model"}},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 1
    assert collection.count() == 1

    result = collection.get(ids=["test.pdf:#/pictures/0"])
    assert result["documents"][0] == "A chart."
    assert result["metadatas"][0]["source"] == "test.pdf"
    assert result["metadatas"][0]["type"] == "picture"
    assert result["metadatas"][0]["element_number"] == 1
    assert result["metadatas"][0]["reference"] == "#/pictures/0"


def test_index_elements_table() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_index_table")

    elements = [
        {
            "element_number": 1,
            "type": "table",
            "reference": "#/tables/0",
            "caption": "",
            "content": {
                "markdown": "| A |\n|---|\n| 1 |",
                "data": {"columns": ["A"], "rows": [["1"]]},
            },
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 1
    assert (
        collection.get(ids=["test.pdf:#/tables/0"])["documents"][0]
        == "| A |\n|---|\n| 1 |"
    )


def test_index_elements_skips_no_content() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_skip")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": None},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 0
    assert collection.count() == 0


def test_index_elements_correct_metadata() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_meta")

    elements = [
        {
            "element_number": 3,
            "type": "picture",
            "reference": "#/pictures/2",
            "caption": "Fig 3",
            "content": {"description": {"text": "Diagram.", "created_by": "m"}},
        }
    ]

    index_elements(elements, "report.pdf", mock_model, collection)
    meta = collection.get(ids=["report.pdf:#/pictures/2"])["metadatas"][0]
    assert meta == {
        "source": "report.pdf",
        "type": "picture",
        "element_number": 3,
        "reference": "#/pictures/2",
    }


def test_index_elements_idempotent() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_idempotent")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "A chart.", "created_by": "m"}},
        }
    ]

    index_elements(elements, "test.pdf", mock_model, collection)
    index_elements(elements, "test.pdf", mock_model, collection)
    assert collection.count() == 1


def test_index_elements_returns_count() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_count")

    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": "Pic 1.", "created_by": "m"}},
        },
        {
            "element_number": 2,
            "type": "table",
            "reference": "#/tables/0",
            "caption": "",
            "content": {
                "markdown": "| A |\n|---|\n| 1 |",
                "data": {"columns": ["A"], "rows": [["1"]]},
            },
        },
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count == 2


def test_index_elements_chunks_long_text() -> None:
    from pipeline.search import index_elements

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    def fake_encode(text: str) -> list[int]:
        # Text with many sentences returns >8000, individual parts return small
        if "Long sentence" in text and text.count(". ") > 5:
            return list(range(9000))
        return list(range(100))

    mock_tokenizer.encode = fake_encode
    mock_model.tokenizer = mock_tokenizer
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_chunk")

    long_text = ". ".join([f"Long sentence {i}" for i in range(100)])
    elements = [
        {
            "element_number": 1,
            "type": "picture",
            "reference": "#/pictures/0",
            "caption": "",
            "content": {"description": {"text": long_text, "created_by": "m"}},
        }
    ]

    count = index_elements(elements, "test.pdf", mock_model, collection)
    assert count >= 2

    all_ids = collection.get()["ids"]
    chunk_ids = [id for id in all_ids if "chunk_" in id]
    assert len(chunk_ids) >= 2
    assert "test.pdf:#/pictures/0:chunk_0" in all_ids
    assert "test.pdf:#/pictures/0:chunk_1" in all_ids
