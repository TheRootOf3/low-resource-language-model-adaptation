"""
Utils functions used in tokenizer and embedding modifications.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch


def sample_decodings(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt: str = "Long, long time ago",
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def compare_model_weights(
    model1: PreTrainedModel,
    model2: PreTrainedModel,
) -> list[tuple[int, str]]:
    """Compare weights of two instances of the same model architecture and return different module names."""

    model1_param_list = list(model1.model.named_parameters())
    model2_param_list = list(model2.model.named_parameters())

    unequal_params = []
    for i, (n1, p1) in enumerate(model1_param_list):
        _, p2 = model2_param_list[i]
        if p1.data.shape != p2.data.shape or not bool(torch.all(p1.data.eq(p2.data))):
            unequal_params.append((i, n1))

    return unequal_params
