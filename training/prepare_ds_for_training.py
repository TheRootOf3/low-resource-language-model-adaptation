"""
WURA dataset pre-processing, tokenisation and grouping.


Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".

Inspired by https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""

import argparse
import logging
from itertools import chain

from datasets import load_dataset, concatenate_datasets, DatasetDict

from utils import get_config_model, get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset dictionary for training."
    )
    parser.add_argument(
        "--wura_dataset_path",
        type=str,
        default=None,
        help="The path to the wura dataset.",
    )
    parser.add_argument(
        "--dolma_dataset_path",
        type=str,
        default=None,
        help="The path to the dolma dataset.",
    )
    parser.add_argument(
        "--wura_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the wura dataset to use.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="The maximum number of tokens to be in the train set. By deafult uses all tokens.",
    )
    parser.add_argument(
        "--english_ratio",
        default=0.5,
        type=float,
        help="The ratio of the english subset in the final dataset.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Path to a tokenizer.",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to a pretrained model or model identifier from huggingface.co/models.",
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
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )

    args = parser.parse_args()

    # Sanity checks

    return args


WURA_VALIDATION_SET_RATIO = 0.1


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples, block_size: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    raw_datasets = load_dataset(
        args.wura_dataset_path,
        args.wura_dataset_config_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    dolma_dataset = (
        load_dataset(
            args.dolma_dataset_path,
            name="v1_6-sample",
            cache_dir=".cache",
            trust_remote_code=True,
        )["train"]
        .shuffle(seed=42)
        .select(range(16 * len(raw_datasets["train"])))
    )

    dolma_dataset = dolma_dataset.rename_column("text", "content")

    logging.info(f"Samples in wura train split: {len(raw_datasets['train'])}")
    logging.info(f"Samples in wura eval split: {len(raw_datasets['validation'])}")

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    wura_column_names = raw_datasets["train"].column_names
    dolma_column_names = dolma_dataset.column_names

    assert (
        "content" in wura_column_names
    ), f"No 'content' column in the dataset. Available columns: {wura_column_names}"

    text_column_name = "content"

    logging.info(f"Using '{text_column_name}' as the column with text content.")

    logging.info("Example samples from wura:")
    logging.info(raw_datasets["train"][:3][text_column_name])

    logging.info("Example samples from dolma:")
    logging.info(dolma_dataset[:3][text_column_name])

    tokenizer = get_tokenizer(
        args.tokenizer_name_or_path,
        args.cache_dir,
    )

    config, _ = get_config_model(
        args.model_name_or_path,
        args.cache_dir,
        load_model=False,
    )

    raw_datasets = raw_datasets.filter(
        lambda x: x[text_column_name] is not None,
        num_proc=args.preprocessing_num_workers,
        desc="Checking for None values in wura.",
    )

    dolma_dataset = dolma_dataset.filter(
        lambda x: x[text_column_name] is not None,
        num_proc=args.preprocessing_num_workers,
        desc="Checking for None values in dolma.",
    )

    tokenized_wura = raw_datasets.map(
        lambda x: tokenizer(x[text_column_name]),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=wura_column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on wura dataset.",
    )

    tokenized_dolma = dolma_dataset.map(
        lambda x: tokenizer(x[text_column_name]),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dolma_column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dolma dataset.",
    )

    wura_train_tokens_count = 0
    cutoff = None
    for i, row in enumerate(tokenized_wura["train"]):
        wura_train_tokens_count += len(row["input_ids"])
        if args.max_tokens is not None and wura_train_tokens_count > args.max_tokens:
            cutoff = i
            break

    if cutoff is not None:
        tokenized_wura["train"] = tokenized_wura["train"].select(range(cutoff))

        tokenized_wura["validation"] = tokenized_wura["validation"].select(
            range(
                int(
                    (
                        len(tokenized_wura["train"])
                        * WURA_VALIDATION_SET_RATIO
                        / (1 - WURA_VALIDATION_SET_RATIO)
                    )
                ),
            )
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logging.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(2048, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(2048, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logging.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    grouped_datasets_wura = tokenized_wura.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping wura texts in chunks of {block_size}",
    )

    grouped_dataset_dolma = tokenized_dolma.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping dolma texts in chunks of {block_size}",
    )

    grouped_dataset_dolma = grouped_dataset_dolma.shuffle(seed=42)

    dolma_train_len = int(
        args.english_ratio
        / (1 - args.english_ratio)
        * len(grouped_datasets_wura["train"])
    )
    dolma_validation_len = int(
        args.english_ratio
        / (1 - args.english_ratio)
        * len(grouped_datasets_wura["validation"])
    )

    logging.info(f"Wura: {grouped_datasets_wura}")
    logging.info(f"Dolma: {grouped_dataset_dolma}")
    logging.info(f"{dolma_train_len}, {dolma_validation_len}")

    grouped_datasets_dolma = DatasetDict(
        {
            "train": grouped_dataset_dolma.select(range(dolma_train_len)),
            "validation": grouped_dataset_dolma.select(
                range(
                    dolma_train_len,
                    dolma_train_len + dolma_validation_len,
                )
            ),
        }
    )

    lm_datasets = DatasetDict(
        {
            "train": concatenate_datasets(
                [
                    grouped_datasets_wura["train"],
                    grouped_datasets_dolma["train"],
                ]
            ).shuffle(seed=42),
            "validation": concatenate_datasets(
                [
                    grouped_datasets_wura["validation"],
                    grouped_datasets_dolma["validation"],
                ]
            ).shuffle(seed=42),
        }
    )

    logging.info(f"Final datasets: {lm_datasets}")

    logging.info(
        f"Wura Train Tokens Count: {block_size * len(grouped_datasets_wura['train'])}"
    )
    logging.info(f"Final Train Tokens Count: {block_size * len(lm_datasets['train'])}")

    # WARNING!
    # Decrease the split number to not allow for a single data file
    # load_dataset infers the dataset file format based on the number
    # of files in the directory LOL
    lm_datasets.save_to_disk(args.output_dir, max_shard_size="100MB")


if __name__ == "__main__":

    main()
