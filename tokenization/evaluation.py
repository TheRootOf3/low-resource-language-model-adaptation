"""
Functions for tokenizer fertility evaluation.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

import time

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from nltk.tokenize import word_tokenize


def get_word_count(line: str) -> int:
    return len(line.split(" "))


def compare_tokenizers_fertility(
    ds_dict: dict[str, Dataset],
    tokenizers_dict: dict[str, PreTrainedTokenizerBase],
    num_of_docs_to_tokenize: int = 10_000,
) -> tuple[dict, dict]:
    c_name = "text" if "text" in ds_dict[list(ds_dict.keys())[0]][0] else "content"

    results = {}
    results_ratios = {}

    for lang, _ in tqdm(ds_dict.items()):
        ds_dict[lang] = ds_dict[lang].shuffle(seed=42)

        tokenizer_results = {}
        tokenizer_results_ratios = {}

        words_count_nltk = sum(
            [
                len(word_tokenize(x))
                for x in ds_dict[lang][:num_of_docs_to_tokenize][c_name]
            ]
        )

        tokenizer_results["words count"] = words_count_nltk

        for tokenizer_name, tokenizer in tokenizers_dict.items():
            t0 = time.time()
            tok_count = sum(
                [
                    len(x)
                    for x in tokenizer(ds_dict[lang][:num_of_docs_to_tokenize][c_name])[
                        "input_ids"
                    ]
                ]
            )
            t1 = time.time()

            tokenizer_results[f"{tokenizer_name} token count"] = tok_count
            tokenizer_results[f"{tokenizer_name} token/words"] = (
                tok_count / words_count_nltk
            )
            tokenizer_results[f"{tokenizer_name} tokenisation time"] = t1 - t0
            tokenizer_results_ratios[tokenizer_name] = tok_count / words_count_nltk

        results[lang] = tokenizer_results
        results_ratios[lang] = tokenizer_results_ratios
    return results, results_ratios
