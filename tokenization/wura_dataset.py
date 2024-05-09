"""
Functions for WURA dataset preparation, pre-processing and upsampling.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from __future__ import annotations
from typing import Optional

import logging

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset


def prepare_wura_dataset(
    dataset_path: str,
    langs: list[str],
    sample: bool = False,
    lang_sample_size: Optional[int] = None,
    combine_headline_content: bool = False,
    num_proc: int = 2,
):
    logging.info(f"Preparing wura dataset.")
    logging.info(f"Using {num_proc} cores.")

    ds_dict = {}
    for lang in langs:
        # load and drop stuff that is not needed
        ds_dict[lang] = load_dataset(
            # path="../../data/wura",
            path=dataset_path,
            name=lang,
            trust_remote_code=True,
            cache_dir="./cache",
        ).remove_columns(["id", "category", "url"])

        # remove empty (None) values
        for split in ["train", "validation"]:
            ds_dict[lang][split] = ds_dict[lang][split].filter(
                lambda x: x["headline"] is not None and x["content"] is not None,
                num_proc=num_proc,
            )

        # combine headline and content if required
        if combine_headline_content:
            logging.info(f"Combining headline and content for {lang}.")
            for split in ["train", "validation"]:
                ds_dict[lang][split] = combine_headline_and_content(
                    ds_dict[lang][split], num_proc=num_proc
                )

    if sample:
        ds_dict = _up_sample_low_res_langs(ds_dict, lang_sample_size)

    return ds_dict


def _up_sample_low_res_langs(
    raw_ds_dict: dict[str, DatasetDict], lang_split_size: int
) -> dict[str, DatasetDict]:
    logging.info(f"Sampling {lang_split_size} samples per language.")

    ds_dict_upsampled = {}

    for lang, ds in raw_ds_dict.items():
        upsampling_ratio = 1
        ds_len = len(ds["train"])
        ds_dict_upsampled[lang] = DatasetDict()

        if lang_split_size < ds_len:
            ds_dict_upsampled[lang]["train"] = (
                ds["train"].shuffle(seed=42).select(range(lang_split_size))
            )
        else:
            # add multiples of the dataset
            concat_tmp = concatenate_datasets(
                [ds["train"] for _ in range(lang_split_size // ds_len)]
            )

            # add the remainder
            ds_dict_upsampled[lang]["train"] = concatenate_datasets(
                [
                    concat_tmp,
                    ds["train"]
                    .shuffle(seed=42)
                    .select(range(lang_split_size - len(concat_tmp))),
                ]
            ).shuffle(seed=42)

        # add validation
        ds_dict_upsampled[lang]["validation"] = ds["validation"]

        upsampling_ratio = len(ds_dict_upsampled[lang]["train"]) / len(ds["train"])
        logging.info(
            f"{'Upsampling' if upsampling_ratio > 1 else 'Downsampling'} ratio for {lang}: {upsampling_ratio}"
        )

    return ds_dict_upsampled


def combine_headline_and_content(ds: Dataset, num_proc: int):
    return ds.map(
        lambda batch: {
            "text": list(
                map(lambda a, b: a + " " + b, batch["headline"], batch["content"])
            )
        },
        remove_columns=["headline", "content"],
        batched=True,
        num_proc=num_proc,
    )
