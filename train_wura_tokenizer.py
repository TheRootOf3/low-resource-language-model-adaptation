"""
Script for training language-dedicated tokenizers on 
language subsets of WURA.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

import argparse
import multiprocessing
import os
import logging

from transformers import AutoTokenizer
from datasets import concatenate_datasets

from tokenization import (
    prepare_wura_dataset,
    batch_sample_generator,
    train_new_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenizer for a chosen language(s) from wura dataset."
    )

    parser.add_argument(
        "--langs",
        type=str,
        required=True,
        help="A comma-separated list of languages from wura dataset to train a tokenizer in.",
    )

    parser.add_argument(
        "--old_tokenizer",
        type=str,
        required=True,
        help="Name of a tokenizer to use a base.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to a directory where to store the trained tokenizer.",
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Name of a tokenizer to use a base.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        help="Sample size.",
    )

    parser.add_argument(
        "--token_count",
        type=int,
        default=None,
        help="Number of tokens in a new tokenizer. Do not provide to keep the size of the original GPT2 tokenizer.",
    )

    args = parser.parse_args()

    # NOTE: Sanity checks which are currently missing

    return args


def main():
    args = parse_args()

    lang_list = args.langs.split(",")

    ds_dict = prepare_wura_dataset(
        "./data/wura",
        lang_list,
        sample=args.sample,
        lang_sample_size=args.sample_size,
        combine_headline_content=True,
        num_proc=multiprocessing.cpu_count(),
    )

    ds = concatenate_datasets([ds["train"] for ds in ds_dict.values()]).shuffle(42)

    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer)

    logging.info(
        f"Using {old_tokenizer.__class__.__name__} and split of wura dataset in {args.langs}."
    )

    training_ds = batch_sample_generator(ds)
    new_tokenizer = train_new_tokenizer(
        old_tokenizer, training_ds, new_token_count=args.token_count
    )

    prompt = "Hello World! This is an example of tokenization."
    print("--- Example Tokenization ---")
    print(
        f"Old tokenizer {args.old_tokenizer}:",
        [old_tokenizer.decode([token]) for token in old_tokenizer(prompt)["input_ids"]],
        old_tokenizer(prompt)["input_ids"],
        sep="\n",
        end="\n\n",
    )
    print(
        f"New tokenizer for {args.langs}:",
        [new_tokenizer.decode([token]) for token in new_tokenizer(prompt)["input_ids"]],
        new_tokenizer(prompt)["input_ids"],
        sep="\n",
    )

    new_tokenizer.save_pretrained(
        os.path.join(
            args.output_dir,
            f"{args.sample_size if args.sample else 'full'}-{args.langs}-{args.old_tokenizer}",
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
