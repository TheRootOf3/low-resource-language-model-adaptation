"""
Utils for model training and dataset processing.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoConfig,
    AutoModelForCausalLM,
)
import torch


def get_dict_from_args_str(args_str: str) -> dict[str, str]:
    """Converts a comma-separated string of key-value pairs into a dictionary.

    Args:
        args_str (str): Comma-separated string of key-value pairs.

    Returns:
        dict[str, str]: Converted dictionary with corresponding key-value pairs.
    """
    args_dict = {}
    for key_val in args_str.split(","):
        assert (
            "=" in key_val
        ), "Beep boop... Wrong format of wandb_args. Make sure it has the following format: key1=value1,key2=value2"
        k, v = key_val.split("=")
        args_dict[k] = v

    assert (
        "project" in args_dict
    ), "Beep boop... Wrong format of wandb_args. Missing value for `project`."

    return args_dict


def get_config_model(
    model_name_or_path: str,
    cache_dir: str,
    trust_remote_code: bool = False,
    load_model: bool = True,
) -> tuple[AutoConfig, AutoModelForCausalLM]:
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    return config, model


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
    )
