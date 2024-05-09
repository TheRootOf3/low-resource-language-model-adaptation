"""
Tokenizer adaptation through token addition.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

import logging

from tokenization import add_tokens, instantiate_tokenizers


def main():

    for lang in ["amh", "eng", "fra", "hau", "ibo", "yor"]:  # Add swa when available
        for k_add in [100, 500, 1000, 2000, 5000, 10000, 15000]:
            new_tokenizer_path = (
                f"./trained_tokenizers/added-opt-{lang}/opt_{k_add}-add_full-{lang}-opt"
            )
            tokenizer_paths = {
                "OPT": "/SAN/intelsys/llm/aszablew/UCL_FYP/raw_tokenizers/opt1.3b",
                "lang": f"/SAN/intelsys/llm/aszablew/UCL_FYP/trained_tokenizers/opt-1.3b-full/full-{lang}-opt1.3b",
            }

            tokenizers = instantiate_tokenizers(tokenizer_paths)
            try:
                add_tokens(
                    base_tokenizer=tokenizers["OPT"],
                    lang_tokenizer=tokenizers["lang"],
                    combined_tokenizer_path=new_tokenizer_path,
                    k_add=k_add,
                )
            except AssertionError as e:
                print(e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
