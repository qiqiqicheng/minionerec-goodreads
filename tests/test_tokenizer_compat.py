import json

from minionerec_goodreads.utils.sft import _tokenizer_compat_kwargs


def test_qwen_extra_special_tokens_list_is_mapped_to_additional_tokens(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}),
        encoding="utf-8",
    )

    kwargs = _tokenizer_compat_kwargs(tmp_path, {"trust_remote_code": True})

    assert kwargs["extra_special_tokens"] == {}
    assert kwargs["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"]


def test_qwen_extra_special_tokens_compat_does_not_clobber_explicit_kwargs(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}),
        encoding="utf-8",
    )

    kwargs = _tokenizer_compat_kwargs(
        tmp_path,
        {
            "extra_special_tokens": {"image_token": "<image>"},
            "additional_special_tokens": ["<custom>"],
        },
    )

    assert kwargs["extra_special_tokens"] == {"image_token": "<image>"}
    assert kwargs["additional_special_tokens"] == ["<custom>"]
