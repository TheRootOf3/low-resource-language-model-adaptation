"""
Utils for tokeniser adaptation.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def instantiate_tokenizers(
    tokenizer_dict: dict[str, str]
) -> dict[str, PreTrainedTokenizerBase]:
    return {k: AutoTokenizer.from_pretrained(v) for k, v in tokenizer_dict.items()}
