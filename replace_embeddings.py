"""
Script for replacing and re-initialising embedding entries of OPT models.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

import os
import json

from transformers import OPTForCausalLM, GPT2TokenizerFast

from modelling import (
    reinitialize_wte_with_average_tokens_,
    compare_model_weights,
    sample_decodings,
)


def replace_embeddings(
    base_tokenizer_path: str,
    tokenizer_path: str,
    old_tokenizer: GPT2TokenizerFast,
    old_model: OPTForCausalLM,
):
    new_tokenizer_path = os.path.join(base_tokenizer_path, tokenizer_path)

    new_tokenizer = GPT2TokenizerFast.from_pretrained(
        new_tokenizer_path, cache_dir=".cache"
    )

    token_ids_to_reinitialize = []

    with open(os.path.join(new_tokenizer_path, "replaced_token_ids.json")) as f:
        token_ids_to_reinitialize = json.load(f)
    assert (
        len(token_ids_to_reinitialize) != 0
    ), "Found no tokens to replace, make sure the new tokenizer path is set correctly!"

    model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir=".cache")

    reinitialize_wte_with_average_tokens_(
        model.model.decoder.embed_tokens,
        token_ids_to_reinitialize,
        old_tokenizer,
        new_tokenizer,
    )

    print(f"1: {sample_decodings(old_tokenizer, old_model)}")
    print(f"2: {sample_decodings(new_tokenizer, model)}")
    print()

    if len(different_modules := compare_model_weights(old_model, model)) == 0:
        print("Models are identical, no differing modules!")
    else:
        print("Models differ in the following modules:")
        for i, name in different_modules:
            print(i, name)

    model.save_pretrained(
        f"models_with_edited_embeddings/{tokenizer_path.split('/')[-1]}"
    )
    new_tokenizer.save_pretrained(
        f"models_with_edited_embeddings/{tokenizer_path.split('/')[-1]}"
    )


def main():
    old_tokenizer = GPT2TokenizerFast.from_pretrained(
        "./raw_tokenizers/opt1.3b", cache_dir=".cache"
    )

    old_model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir=".cache")
    print(old_model.model)

    base_tokenizer_path = "./trained_tokenizers"

    for lang in ["hau", "ibo", "yor", "amh"]:
        for tokenizer_path in [
            f"replaced-opt-{lang}/opt_100-replace_full-{lang}-opt",
            f"replaced-opt-{lang}/opt_2000-replace_full-{lang}-opt",
        ]:
            print(f"Processing: {tokenizer_path}...")
            replace_embeddings(
                base_tokenizer_path=base_tokenizer_path,
                tokenizer_path=tokenizer_path,
                old_tokenizer=old_tokenizer,
                old_model=old_model,
            )


if __name__ == "__main__":
    main()
