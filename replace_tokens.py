"""
Tokenizer adaptation through token replacement.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

import logging

from tokenization import replace_vocab_opt


def main():
    base_tokenizer_path = "/SAN/intelsys/llm/aszablew/UCL_FYP/raw_tokenizers/opt1.3b"

    for lang in ["amh", "eng", "fra", "hau", "ibo", "yor"]:
        lang_tokenizer_path = f"/SAN/intelsys/llm/aszablew/UCL_FYP/trained_tokenizers/opt-1.3b-full/full-{lang}-opt1.3b"

        for k_replace in [100, 500, 1000, 2000, 5000, 10000, 15000, -1]:

            new_tokenizer_path = f"./trained_tokenizers/replaced-opt-{lang}/opt_{k_replace if k_replace != -1 else 'all'}-replace_full-{lang}-opt"
            try:
                replace_vocab_opt(
                    base_tokenizer_path=base_tokenizer_path,
                    lang_tokenizer_path=lang_tokenizer_path,
                    combined_tokenizer_path=new_tokenizer_path,
                    k_replace=k_replace,
                )
            except AssertionError as e:
                print(e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
