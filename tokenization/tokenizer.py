"""
Token addition and replacement functions.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from typing import Optional

import os
import json
import shutil
import logging
from typing import Generator

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm


def batch_sample_generator(ds: Dataset) -> Generator:
    for start_idx in range(0, int(len(ds)), 1000):
        samples = ds[start_idx : start_idx + 1000]
        yield samples["text"] if "text" in samples else samples["content"]


def train_new_tokenizer(
    old_tokenizer: PreTrainedTokenizerBase,
    training_ds: Dataset,
    new_token_count: Optional[int] = None,
) -> PreTrainedTokenizerBase:
    if new_token_count is None:
        # new_token_count = 50257  # GPT2 Tokenizer
        new_token_count = old_tokenizer.vocab_size

    return old_tokenizer.train_new_from_iterator(training_ds, new_token_count)


def bytes_to_unicode() -> dict[int, str]:
    """
    From: https://colab.research.google.com/drive/15tMASZ0NLm8bnxkM4uXCRgdzznSpbp9L?usp=sharing#scrollTo=v5BPJsHetnPP
    From: roudimit, https://github.com/huggingface/tokenizers/issues/1162#issuecomment-1437435152
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def add_tokens(
    base_tokenizer: PreTrainedTokenizerBase,
    lang_tokenizer: PreTrainedTokenizerBase,
    combined_tokenizer_path: str,
    k_add: int,
) -> None:

    lang_unique_tokens = set(lang_tokenizer.vocab.keys()).difference(
        base_tokenizer.vocab.keys()
    )

    # # Remove the continuation character 'Ġ' from all "continuation" tokens
    # lang_unique_tokens = set(
    #     map(lambda x: x if x[0] != "Ġ" else x[1:], lang_tokenizer.vocab.keys())
    # ).difference(base_tokenizer.vocab.keys())

    assert (
        len(lang_unique_tokens) >= k_add
    ), f"There is not enough new tokens ({len(lang_unique_tokens)}) to add the requested k_add={k_add} tokens. Choose smaller k_add."

    # Select top k tokens from lang tokenizer to add to the base tokenizer vocabulary
    lang_tokens_to_replace = filter_and_sort_unique_tokens(
        lang_unique_tokens, lang_tokenizer.vocab
    )

    # turn BPE-decoded tokens into unicode-decoded characters
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    tokens_to_add = []
    for x in lang_tokens_to_replace:
        try:
            tokens_to_add.append(
                bytearray([byte_decoder[c] for c in x[0]]).decode("utf-8")
            )
        except UnicodeDecodeError as e:
            print(e, x)

    assert (
        len(tokens_to_add) >= k_add
    ), f"A maximum of {len(tokens_to_add)} can be added. Decrease k_add."

    logging.info(
        f"A {k_add} out of available {len(tokens_to_add)} tokens from the lang tokenizer will be added to the base tokenizer!"
    )

    added_tokens = base_tokenizer.add_tokens(
        tokens_to_add[:k_add],
        special_tokens=False,
    )
    logging.info(f"Added tokens: {added_tokens}")

    base_tokenizer.save_pretrained(combined_tokenizer_path)


def filter_and_sort_unique_tokens(
    unique_token_set: set[str], vocab_dict: dict[str, int]
) -> list[str, int]:
    tmp_dict = dict(filter(lambda k: k[0] in unique_token_set, vocab_dict.items()))

    return [
        (k, v)
        for k, v in sorted(
            tmp_dict.items(),
            key=lambda x: x[1],
        )
    ]


def replace_vocab_opt(
    base_tokenizer_path: str,
    lang_tokenizer_path: str,
    combined_tokenizer_path: str,
    k_replace: int = -1,
) -> None:
    # Read vocab and merges
    with open(
        os.path.join(base_tokenizer_path, "vocab.json"), encoding="utf-8"
    ) as vocab_handle:
        vocab_base = json.load(vocab_handle)

    with open(
        os.path.join(lang_tokenizer_path, "vocab.json"), encoding="utf-8"
    ) as vocab_handle:
        vocab_lang = json.load(vocab_handle)

    # Read merges
    with open(
        os.path.join(base_tokenizer_path, "merges.txt"), encoding="utf-8"
    ) as merges_handle:
        merges_base = [line.rstrip() for line in merges_handle.readlines()]

    with open(
        os.path.join(lang_tokenizer_path, "merges.txt"), encoding="utf-8"
    ) as merges_handle:
        merges_lang = [line.rstrip() for line in merges_handle.readlines()]

    vocab_new, merges_new, replace_tokens_ids = replace_vocab_opt_bpe(
        vocab_base=vocab_base,
        vocab_lang=vocab_lang,
        merges_base=merges_base,
        merges_lang=merges_lang,
        k_replace=k_replace,
    )

    os.makedirs(combined_tokenizer_path)
    with open(
        os.path.join(combined_tokenizer_path, "vocab.json"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps(vocab_new, ensure_ascii=False) + "\n")

    with open(
        os.path.join(combined_tokenizer_path, "merges.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines([line + "\n" for line in merges_new])

    shutil.copyfile(
        os.path.join(base_tokenizer_path, "tokenizer_config.json"),
        os.path.join(combined_tokenizer_path, "tokenizer_config.json"),
    )
    shutil.copyfile(
        os.path.join(base_tokenizer_path, "special_tokens_map.json"),
        os.path.join(combined_tokenizer_path, "special_tokens_map.json"),
    )

    # Save replace token ids to a file
    with open(
        os.path.join(combined_tokenizer_path, "replaced_token_ids.json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(replace_tokens_ids, ensure_ascii=False) + "\n")


def replace_vocab_opt_bpe(
    vocab_base: dict[str, int],
    vocab_lang: dict[str, int],
    merges_base: list[str],
    merges_lang: list[str],
    k_replace: int,
) -> tuple[dict[str, int], list[str]]:

    # compute set of all non-final tokens in the base tokenizer
    # final tokens are those, which are not parts of any merge rule
    not_final_base_tokens = set()
    for line in merges_base[1:]:  # skip the first line which contain the version number
        not_final_base_tokens = not_final_base_tokens.union(
            set(line.rstrip().split(" "))
        )

    # calculate set of all final tokens in the base tokenizer
    base_final_unique_tokens = (
        set(vocab_base.keys())  # all tokens from the base tokenizer
        .difference(
            set(vocab_lang.keys())
        )  # unique tokens which are only in the base tokenizer
        .difference(
            not_final_base_tokens
        )  # unique and final (not merge-able) tokens from the base tokenizer
    )

    # calculate set of new unique tokens in the transplant tokenizer
    lang_unique_tokens = set(vocab_lang.keys()).difference(vocab_base.keys())
    logging.info(f"There are {len(lang_unique_tokens)} new unique tokens!")
    logging.info(f"There are {len(base_final_unique_tokens)} old unique, final tokens!")

    k = min(len(lang_unique_tokens), len(base_final_unique_tokens))

    assert (
        k_replace <= k
    ), f"Can only replace up to {k} tokens. Choose smaller k_replace."

    if k_replace == -1:
        logging.info(f"Replacing the maximum possible number of tokens: ({k}).")
    elif k >= k_replace:
        k = k_replace
        logging.info(f"Using capped replacement size set by k_replace={k_replace}.")

    logging.info(
        f"Plan: Replace {k} least frequent old unique tokens with {k} most frequent new unique tokens."
    )

    # Select top k tokens from lang tokenizer and last k final unique tokens from base to replace
    lang_tokens_to_replace = filter_and_sort_unique_tokens(
        lang_unique_tokens, vocab_lang
    )[:k]
    base_tokens_to_replace = filter_and_sort_unique_tokens(
        base_final_unique_tokens, vocab_base
    )[-k:]

    # print(lang_tokens_to_replace, base_tokens_to_replace)

    merges_base_for_searching = list(map(lambda x: x.replace(" ", ""), merges_base))
    merges_lang_for_searching = list(map(lambda x: x.replace(" ", ""), merges_lang))

    replace_token_ids = []
    for i in tqdm(range(k)):
        old_token, old_idx = base_tokens_to_replace[i]
        new_token, new_idx = lang_tokens_to_replace[i]

        # replace vocab
        del vocab_base[old_token]
        vocab_base[new_token] = old_idx
        replace_token_ids.append(old_idx)
        if old_token in merges_base_for_searching:
            tmp_id1 = merges_base_for_searching.index(old_token)
            tmp_id2 = merges_lang_for_searching.index(new_token)

            assert (
                merges_base[tmp_id1].replace(" ", "").rstrip() == old_token
                and merges_lang[tmp_id2].replace(" ", "").rstrip() == new_token
            )
            merges_base[tmp_id1] = merges_lang[tmp_id2]

        else:
            logging.info(
                f"removing a 'special' token addded at the end: {old_token}. Adding new merge rule at the end of the file."
            )
            tmp_id = merges_lang_for_searching.index(new_token)

            assert merges_lang[tmp_id].replace(" ", "").rstrip() == new_token
            merges_base.append(merges_lang[tmp_id])

    logging.info(f"Replaced {k} tokens!")

    vocab_new = {
        k: v
        for k, v in sorted(
            vocab_base.items(),
            key=lambda x: x[1],
        )
    }
    return vocab_new, merges_base, replace_token_ids
