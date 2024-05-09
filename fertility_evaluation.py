"""
Script for running experiments of fertility evaluation
of tokenizers using the language subsets of WURA.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

import logging
import os

from tokenization import compare_tokenizers_fertility, instantiate_tokenizers
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt


def save_and_display_results(
    results: dict,
    results_ratios: dict,
    experiment_name: str,
    show: bool = False,
):
    results_pd = pd.DataFrame.from_dict(results, "index")
    results_ratios_pd = pd.DataFrame.from_dict(results_ratios, "index")

    os.makedirs(f"fertility_analysis/{experiment_name}")

    results_pd.to_csv(f"fertility_analysis/{experiment_name}/full_results.csv")
    results_ratios_pd.to_csv(f"fertility_analysis/{experiment_name}/ratio.csv")

    print("Results")
    print(results_pd)
    print("Fertility per tokenizer")
    print(results_ratios_pd)

    fig, ax = plt.subplots()

    results_ratios_pd.plot.bar(alpha=0.85, zorder=2, ax=ax)
    ax.set_title("Tokenizer Fertility (↓)")
    ax.set_xlabel("Tokenized Subset of Wura")
    ax.set_ylabel("Fertility")
    ax.grid(axis="y", color="k", alpha=0.2, zorder=1)

    plt.savefig(f"fertility_analysis/{experiment_name}/plot.png", dpi=300)

    if show:
        plt.show()


def main():
    # for lang in ["hau", "ibo", "yor", "amh"]:
    for lang in ["amh"]:

        ds_dict = {}

        # for ds_lang in ["eng", "fra", "hau", "ibo", "swa", "yor", "amh"]:
        for ds_lang in ["eng", lang]:
            # load and drop stuff that is not needed
            ds_dict[ds_lang] = load_dataset(
                path="./data/wura",
                name=ds_lang,
                split="validation",
                trust_remote_code=True,
                cache_dir=".cache",
            ).remove_columns(["id", "category", "url", "headline"])

            logging.info(f"Loaded validation split for {ds_lang}.")

            ds_dict[ds_lang] = ds_dict[ds_lang].filter(
                lambda x: x["content"] is not None, desc="Filtering for None values."
            )

        tokenizer_paths = {
            # "GPT2 BPE": "./raw_tokenizers/gpt2",
            "OPT BPE": "./raw_tokenizers/opt1.3b",
        }

        for k_replace in [100, 500, 1000, 2000, 5000, 10000, 15000]:
            tokenizer_paths[f"opt-{lang}-replaced-{k_replace}"] = (
                f"./trained_tokenizers/replaced-opt-{lang}/opt_{k_replace}-replace_full-{lang}-opt"
            )

        for k_add in [100, 500, 1000, 2000, 5000, 10000, 15000]:
            tokenizer_paths[f"opt-{lang}-added-{k_add}"] = (
                f"./trained_tokenizers/added-opt-{lang}/opt_{k_add}-add_full-{lang}-opt"
            )

        tokenizer_paths[f"{lang}"] = (
            f"./trained_tokenizers/opt-1.3b-full/full-{lang}-opt1.3b"
        )

        tokenizers = instantiate_tokenizers(tokenizer_paths)
        logging.info(f"Loaded tokenizers: {list(tokenizer_paths.keys())}.")

        experiment_name = f"2_opt_{lang}_added_and_replaced_tokens"

        results, results_ratios = compare_tokenizers_fertility(ds_dict, tokenizers)
        save_and_display_results(results, results_ratios, experiment_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
