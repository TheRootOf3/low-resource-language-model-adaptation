"""
Aya Dataset pre-processing and tokenisation.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".

Inspired by https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""

import argparse
import logging
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset dictionary for training."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="The language from the dataset to use.",
    )
    parser.add_argument(
        "--english_ratio",
        default=0.25,
        type=float,
        help="The ratio of the english subset in the final dataset.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to cache downloaded weights and datasets.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the processed datasets.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    args = parser.parse_args()

    # Sanity checks

    return args


def get_tokenizer(
    tokenizer_name_or_path: str,
    cache_dir: str,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        padding_side="right",
        truncation_side="right",
    )


def combine_inputs_and_targets(ds: Dataset):
    return ds.map(
        lambda batch: {
            "content": list(
                map(lambda a, b: a + " " + b, batch["inputs"], batch["targets"])
            )
        },
        remove_columns=["inputs", "targets"],
        batched=True,
    )


def tokenize_and_add_labels(
    x: Union[str, list[str]], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[int]]:
    output_dict = tokenizer(
        x,
        padding="max_length",
        truncation=True,
        max_length=2048,
    )
    output_dict["labels"] = output_dict["input_ids"].copy()

    return output_dict


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    raw_datasets = load_dataset(
        args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir
    )

    if "eval" in raw_datasets.keys():
        raw_datasets["validation"] = raw_datasets["eval"]

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
        )
    # Preprocessing the datasets.
    # First we tokenize all the texts.

    eng_datasets = raw_datasets.filter(
        lambda x: x["language_code"] == "eng",
        num_proc=args.preprocessing_num_workers,
        desc=f"Filtering for language eng.",
    )
    logging.info(
        f"Eng dataset lengths: train - {len(eng_datasets['train'])}, val - {len(eng_datasets['validation'])}."
    )

    raw_datasets = raw_datasets.filter(
        lambda x: x["language_code"] == args.lang,
        num_proc=args.preprocessing_num_workers,
        desc=f"Filtering for language {args.lang}",
    )
    logging.info(
        f"Language dataset lengths: train - {len(raw_datasets['train'])}, val - {len(raw_datasets['validation'])}."
    )

    raw_datasets["validation"] = combine_inputs_and_targets(raw_datasets["validation"])
    raw_datasets["train"] = combine_inputs_and_targets(raw_datasets["train"])

    eng_datasets["validation"] = combine_inputs_and_targets(eng_datasets["validation"])
    eng_datasets["train"] = combine_inputs_and_targets(eng_datasets["train"])

    del raw_datasets["test"]
    del eng_datasets["test"]

    column_names = raw_datasets["train"].column_names

    assert (
        "content" in column_names
    ), f"No 'content' column in the dataset. Available columns: {column_names}"

    text_column_name = "content"

    logging.info(f"Using '{text_column_name}' as the column with text content.")

    logging.info("Example samples from the eng dataset:")
    logging.info(eng_datasets["train"][:3][text_column_name])

    logging.info("Example samples from the lang dataset:")
    logging.info(raw_datasets["train"][:3][text_column_name])

    tokenizer = get_tokenizer(
        args.tokenizer_name_or_path,
        args.cache_dir,
    )

    raw_datasets = raw_datasets.filter(
        lambda x: x[text_column_name] is not None,
        num_proc=args.preprocessing_num_workers,
    )

    eng_datasets = eng_datasets.filter(
        lambda x: x[text_column_name] is not None,
        num_proc=args.preprocessing_num_workers,
    )

    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_add_labels(
            x[text_column_name],
            tokenizer,
        ),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on lang dataset",
    )

    eng_tokenized_datasets = eng_datasets.map(
        lambda x: tokenize_and_add_labels(
            x[text_column_name],
            tokenizer,
        ),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on eng dataset",
    )

    logging.info(f"Lang: {tokenized_datasets}")

    eng_train_len = int(
        args.english_ratio / (1 - args.english_ratio) * len(tokenized_datasets["train"])
    )
    eng_validation_len = int(
        args.english_ratio
        / (1 - args.english_ratio)
        * len(tokenized_datasets["validation"])
    )

    # If english subset too small, double its size by repeating.
    if eng_train_len > len(eng_tokenized_datasets["train"]):
        eng_tokenized_datasets["train"] = concatenate_datasets(
            [
                eng_tokenized_datasets["train"],
                eng_tokenized_datasets["train"],
                eng_tokenized_datasets["train"],
                eng_tokenized_datasets["train"],
            ]
        )
    if eng_train_len > len(eng_tokenized_datasets["validation"]):
        eng_tokenized_datasets["validation"] = concatenate_datasets(
            [
                eng_tokenized_datasets["validation"],
                eng_tokenized_datasets["validation"],
                eng_tokenized_datasets["validation"],
                eng_tokenized_datasets["validation"],
            ]
        )

    ift_datasets = DatasetDict(
        {
            "train": concatenate_datasets(
                [
                    tokenized_datasets["train"],
                    eng_tokenized_datasets["train"]
                    .shuffle(seed=42)
                    .select(
                        range(eng_train_len),
                    ),
                ]
            ).shuffle(seed=42),
            "validation": concatenate_datasets(
                [
                    tokenized_datasets["validation"],
                    eng_tokenized_datasets["validation"]
                    .shuffle(seed=42)
                    .select(
                        range(
                            eng_validation_len,
                        ),
                    ),
                ]
            ).shuffle(seed=42),
        }
    )

    logging.info(f"Final datasets: {ift_datasets}")

    # WARNING!
    # Decrease the split number to not allow for a single data file
    # load_dataset infers the dataset file format based on the number
    # of files in the directory LOL
    ift_datasets.save_to_disk(args.output_dir, max_shard_size="100MB")


if __name__ == "__main__":

    main()
