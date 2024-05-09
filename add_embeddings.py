"""
Script for extending and re-initialising embedding entries of OPT models.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

import os

from transformers import OPTForCausalLM, GPT2TokenizerFast

from modelling import (
    compare_model_weights,
    sample_decodings,
    reinitialize_wte_with_average_tokens_,
)


def add_embeddings(
    base_tokenizer_path: str,
    tokenizer_path: str,
    old_tokenizer: GPT2TokenizerFast,
    old_model: OPTForCausalLM,
):
    new_tokenizer_path = os.path.join(base_tokenizer_path, tokenizer_path)

    new_tokenizer = GPT2TokenizerFast.from_pretrained(
        new_tokenizer_path, cache_dir=".cache"
    )

    model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir=".cache")

    print(f"Original embedding size: {model.model.decoder.embed_tokens}.")
    print(f"Length of the new tokenizer: {len(new_tokenizer)}.")

    model.model.decoder.embed_tokens = model.resize_token_embeddings(
        new_num_tokens=len(new_tokenizer), pad_to_multiple_of=128
    )
    model.model.decoder.embed_tokens.padding_idx = 1

    tokens_to_initialize = [
        t_id
        for t_id in new_tokenizer.get_added_vocab().values()
        if t_id >= new_tokenizer.vocab_size
    ]

    reinitialize_wte_with_average_tokens_(
        model.model.decoder.embed_tokens,
        tokens_to_initialize,
        old_tokenizer,
        new_tokenizer,
    )

    print(f"Updated embedding size: {model.model.decoder.embed_tokens}.")

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
            f"added-opt-{lang}/opt_100-add_full-{lang}-opt",
            f"added-opt-{lang}/opt_2000-add_full-{lang}-opt",
        ]:
            print(f"Processing: {tokenizer_path}...")
            add_embeddings(
                base_tokenizer_path=base_tokenizer_path,
                tokenizer_path=tokenizer_path,
                old_tokenizer=old_tokenizer,
                old_model=old_model,
            )


if __name__ == "__main__":
    main()
