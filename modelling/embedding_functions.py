"""
Embedding re-initialisation functions used for token replacement and addition.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from typing import Callable

from torch import nn
import torch

from transformers import PreTrainedTokenizerBase


def freeze_module_(module_to_freeze: nn.Module) -> None:
    module_to_freeze.requires_grad_(False)


def unfreeze_module_(module_to_unfreeze: nn.Module) -> None:
    module_to_unfreeze.requires_grad_()


def _reinitialize_wte_entries_(
    wte: nn.Module,
    token_ids_to_reinitialize: list[int],
    reinitialization_function: Callable,
) -> None:
    _, embedding_dim = wte.weight.shape

    freeze_module_(wte)
    for row in token_ids_to_reinitialize:
        wte.weight[row] = reinitialization_function(row, embedding_dim)
    unfreeze_module_(wte)


def reinitialize_wte_with_zeros_(
    wte: nn.Module,
    token_ids_to_reinitialize: list[int],
) -> None:
    _reinitialize_wte_entries_(
        wte=wte,
        token_ids_to_reinitialize=token_ids_to_reinitialize,
        reinitialization_function=lambda _, embedding_dim: torch.zeros(
            1, embedding_dim
        ),
    )


def reinitialize_wte_entries_xavier_(
    wte: nn.Module,
    token_ids_to_reinitialize: list[int],
) -> None:
    _reinitialize_wte_entries_(
        wte=wte,
        token_ids_to_reinitialize=token_ids_to_reinitialize,
        reinitialization_function=lambda row, _: torch.nn.init.xavier_uniform_(
            wte[row]
        ),
    )


def reinitialize_wte_with_average_tokens_(
    wte: nn.Module,
    token_ids_to_reinitialize: list[int],
    old_tokenizer: PreTrainedTokenizerBase,
    new_tokenizer: PreTrainedTokenizerBase,
) -> None:
    _reinitialize_wte_entries_(
        wte=wte,
        token_ids_to_reinitialize=token_ids_to_reinitialize,
        reinitialization_function=lambda row, _: torch.mean(
            wte.weight[
                old_tokenizer(
                    new_tokenizer.decode([row]),
                    add_special_tokens=False,
                )["input_ids"]
            ],
            dim=0,
        ),
    )
