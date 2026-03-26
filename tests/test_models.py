"""Tests for the models module."""

from unittest.mock import MagicMock, patch

import torch


@patch("pipeline.models.AutoModelForImageTextToText")
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


@patch("pipeline.models.AutoModelForImageTextToText")
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


def test_generate_response_trims_and_decodes() -> None:
    from pipeline.models import generate_response

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_processor.decode.return_value = "Generated text."

    conversation = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    result = generate_response(
        conversation, mock_processor, mock_model, max_new_tokens=512
    )

    # Verify apply_chat_template kwargs
    call_kwargs = mock_processor.apply_chat_template.call_args[1]
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["return_dict"] is True
    assert call_kwargs["return_tensors"] == "pt"

    # Verify generate called with max_new_tokens
    _, gen_kwargs = mock_model.generate.call_args
    assert gen_kwargs["max_new_tokens"] == 512

    # Verify trimming: input had 3 tokens, output has 5, so decoded tokens are [4, 5]
    decoded_tensor = mock_processor.decode.call_args[0][0]
    assert torch.equal(decoded_tensor, torch.tensor([4, 5]))
    assert mock_processor.decode.call_args[1]["skip_special_tokens"] is True

    assert result == "Generated text."


def test_generate_response_empty_output() -> None:
    from pipeline.models import generate_response

    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    input_ids = torch.tensor([[1, 2]])
    mock_processor.apply_chat_template.return_value = MagicMock()
    mock_processor.apply_chat_template.return_value.to.return_value = {
        "input_ids": input_ids
    }

    # Model returns same length as input (no new tokens)
    mock_model.generate.return_value = torch.tensor([[1, 2]])
    mock_processor.decode.return_value = ""

    conversation = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    result = generate_response(conversation, mock_processor, mock_model)
    assert result == ""
