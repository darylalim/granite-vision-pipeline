"""Tests for the models module."""

from unittest.mock import MagicMock, patch


@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_load_vision_model_loads_and_moves_to_device(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import _load_vision_model

    processor, model = _load_vision_model("some/repo", device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with("some/repo")
    mock_model_cls.from_pretrained.assert_called_once_with("some/repo")
    mock_model_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")
    assert processor is mock_processor_cls.from_pretrained.return_value
    assert model is mock_model_cls.from_pretrained.return_value.to.return_value


@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_granite_vision_model_uses_correct_repo(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import create_granite_vision_model

    create_granite_vision_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-vision-3.3-2b"
    )


@patch("pipeline.models.AutoModelForVision2Seq")
@patch("pipeline.models.AutoProcessor")
def test_create_doctags_model_uses_correct_repo(
    mock_processor_cls: MagicMock,
    mock_model_cls: MagicMock,
) -> None:
    from pipeline.models import create_doctags_model

    create_doctags_model(device="cpu")

    mock_processor_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
    mock_model_cls.from_pretrained.assert_called_once_with(
        "ibm-granite/granite-docling-258M"
    )
