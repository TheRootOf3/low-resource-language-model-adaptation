"""
LLM zero-shot evaluation script. Used for generating model responses to 
benchmark tasks.

Adapted and optimized from https://github.com/JessicaOjo/LLM-African-eval
by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 
"""

from __future__ import annotations
import os
import argparse
import time

import glob
import json
import pandas as pd
from pathlib import Path
import torch.utils
import torch.utils.data
from transformers import GenerationConfig, GPT2TokenizerFast, OPTForCausalLM
import torch
from tqdm import tqdm
import numpy as np


def load_model(model_name, device):

    model = OPTForCausalLM.from_pretrained(
        model_name,
        cache_dir="../.cache",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # actually slows it down!
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_name,
        cache_dir="../.cache",
        padding_side="left",
    )

    model = model.to(device)

    return model, tokenizer


# script from Nikita Vassilyev and Alex Pejovic
def prompt_llm(
    model: OPTForCausalLM,
    tokenizer: GPT2TokenizerFast,
    messages: list[str],
    device,
    # repetition_penalty: float = 1.176,
    repetition_penalty: float = 1.5,
    num_beams: int = 1,
    max_new_tokens: int = 256,
):

    generation_config = GenerationConfig(
        ### temperature, top_p, and top_k are not needed since we are using 1 beam
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    # print(messages)
    inputs = tokenizer(
        messages,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=1024,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_ids),
        batch_size=32,
        shuffle=False,
    )

    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    generations = []
    for idx, inputs in enumerate(dataloader):
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs[0],
                generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
        print(f"generation progress: {idx}/{len(dataloader)}... batch time: {dt:.2f}s")

        generations.extend(output)

    output = tokenizer.batch_decode(generations, skip_special_tokens=True)
    print("Message:", messages[0], sep="\n")
    print("Entire output:", output[0], sep="\n")
    output = [output[i][len(messages[i]) :] for i in range(len(output))]
    print("Cropped output:", output[0], sep="\n")
    return output


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getlabel_string(filename):
    with open(filename) as f:
        label_list = f.read().splitlines()
    label_string = label_list[0]
    for i, value in enumerate(label_list[:-2], 1):
        label_string += ", " + label_list[i]

    label_string += " or " + label_list[-1]

    return label_string, label_list


def get_language(files, senti=False, mt=False):

    if senti:
        lang = sorted([i.split("/")[-2] for i in files])
        languages = [
            "Amharic",
            "Hausa",
            "Igbo",
            "Yoruba",
        ]
        return dict(zip(lang, languages))
    if mt:
        languages = [
            "Amharic",
            "Hausa",
            "Igbo",
            "Yoruba",
        ]
        lang = [i.split("/")[-2].split("-")[1] for i in files]

        return dict(zip(lang, languages))


def sentiment(model_pipeline, tokenizer, output_dir, device, langs):
    """Identifies tweet sentiments for different languages"""
    create_dir(output_dir)

    files = glob.glob(
        "data_repos/afrisent-semeval-2023//data/**/test.tsv", recursive=True
    )

    assert len(files) != 0

    files = list(filter(lambda x: sum([ll in x for ll in langs]) > 0, files))

    languages = get_language(files, senti=True)
    label = "{{Neutral, Positive or Negative}}"

    for file in tqdm(files):

        df = pd.read_csv(file, sep="\t", header=0)
        language = file.split("/")[-2]
        language = languages[language]
        print(f"\nLanguage: {language}, using file: {file}")
        print(df.head())

        df["prompts"] = df["tweet"].map(
            lambda x: f'Does this {language} statement; "{x}" have a {label} sentiment? Labels only '
        )

        print(f"Avg length of prompts", np.mean(df["prompts"].map(lambda x: len(x))))

        responses = prompt_llm(model_pipeline, tokenizer, list(df["prompts"]), device)

        df["opt"] = responses
        df.to_csv(output_dir + language + ".tsv", sep="\t")


def news_classification(model_pipeline, tokenizer, output_dir, device, langs):
    create_dir(output_dir)
    files = glob.glob("data_repos/masakhane-news/data/**/test.tsv", recursive=True)
    assert len(files) != 0

    files = list(filter(lambda x: sum([ll in x for ll in langs]) > 0, files))

    prompt_prefix = "Is this an article regarding {{"
    prompt_suffix = "}}? "

    for file in tqdm(files):
        file_path = Path(file)
        df = pd.read_csv(file, sep="\t")
        label_string, label_list = getlabel_string(
            Path(f"{file_path.parent}/labels.txt")
        )
        lang = file.split("/")[-2]
        print(f"\nLanguage: {lang}, using file: {file}")
        print(df.head())

        df["prompts"] = df.apply(
            lambda x: "Labels only. "
            + prompt_prefix
            + label_string
            + prompt_suffix
            + "\nArticle: "
            + " ".join(f"{x['headline']} {x['text']}".split()[:100])
            + "\nLabel:",
            axis=1,
        )

        responses = prompt_llm(model_pipeline, tokenizer, list(df["prompts"]), device)

        df["opt"] = responses
        df.to_csv(output_dir + lang + ".tsv", sep="\t")


def cross_lingual_qa(model_pipeline, tokenizer, output_dir, device, langs, pivot=False):
    create_dir(output_dir)

    for language in tqdm(langs):
        print(language)
        gold_passages = glob.glob(
            f"data_repos/afriqa/data/gold_passages/{language}/*test.json"
        )
        assert len(gold_passages) != 0
        gp_df = pd.read_json(gold_passages[0], lines=True)
        print(
            f"\nLanguage: {language}, using file: data_repos/afriqa/data/gold_passages/{language}/*test.json"
        )
        print(gp_df.head())

        pivot_lang = "French" if gold_passages[0].split(".")[-3] == "fr" else "English"
        prompt_query = f"Use the following pieces of context to answer the provided question. If you don't know the answer, \
just say that you don't know, don't try to make up an answer. Provide the answer with the least number of \
words possible. Provide the answer only. Provide answer in {pivot_lang}. Do not repeat the question"

        gp_df["prompt"] = gp_df.apply(
            lambda x: (
                prompt_query
                + "\n"
                + f"Context: {x['context']}"
                + "\n"
                + (
                    f"Question: {x['question_translated']}"
                    if pivot
                    else f"Question: {x['question_lang']}"
                )
                + "\nAnswer:"
            ),
            axis=1,
        )

        responses = prompt_llm(model_pipeline, tokenizer, list(gp_df["prompt"]), device)

        gp_df["opt"] = responses
        gp_df.to_csv(output_dir + language + ".tsv", sep="\t")


def machine_translation(
    model_pipeline, tokenizer, output_dir, device, langs, reverse=False
):
    create_dir(output_dir)

    files = glob.glob("data_repos/lafand-mt/data/tsv_files/**/test.tsv", recursive=True)
    assert len(files) != 0
    files = list(filter(lambda x: sum([ll in x for ll in langs]) > 0, files))

    languages = get_language(files, mt=True)

    for file in tqdm(files):
        df = pd.read_csv(file, sep="\t", header=0)
        pivot_lang_abv, target_lang_abv = (
            file.split("/")[-2].split("-")[0],
            file.split("/")[-2].split("-")[1],
        )
        target_lang = [v for k, v in languages.items() if k == target_lang_abv][0]
        pivot_lang = "English" if pivot_lang_abv == "en" else "French"

        print(f"\nLanguage: {target_lang}, using file: {file}")
        print(df.head())

        if not reverse:
            prompt_query = f"Translate the {pivot_lang} sentence below to {target_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"
            df["prompt"] = df.apply(
                lambda x: (
                    prompt_query + "\n" + x["en"]
                    if pivot_lang_abv == "en"
                    else x["fr"] + " "
                ),
                axis=1,
            )
        else:
            prompt_query = f"Translate the {target_lang} sentence below to {pivot_lang}. Return the translated sentence only. If you cannot translate the sentence simply say you don't know"
            df["prompt"] = df.apply(
                lambda x: prompt_query + "\n" + x[target_lang_abv] + " ",
                axis=1,
            )

        responses = prompt_llm(model_pipeline, tokenizer, list(df["prompt"]), device)

        df["opt"] = responses
        if reverse:
            df.to_csv(
                output_dir + f"{target_lang_abv}-{pivot_lang_abv}" + ".tsv", sep="\t"
            )
        else:
            df.to_csv(
                output_dir + f"{pivot_lang_abv}-{target_lang_abv}" + ".tsv", sep="\t"
            )


def named_entity_recognition(model_pipeline, tokenizer, output_dir, device, langs):
    create_dir(output_dir)

    prompt_query = "Named entities refers to names of location, organisation and personal name. \n\
For example, 'David is an employee of Amazon and he is visiting New York next week to see Esther' will be \n\
PERSON: David $ ORGANIZATION: Amazon $ LOCATION: New York $ PERSON: Esther \n\n\
List all the named entities in the passage above using $ as separator. Return only the output"

    files = glob.glob(
        "data_repos/masakhane-ner/xtreme-up/MasakhaNER-X/test/*.jsonl", recursive=True
    )
    assert len(files) != 0

    langs_ner_map = {
        "ibo": "ig.jsonl",
        "hau": "ha.jsonl",
        "yor": "yo.jsonl",
        "amh": "am.jsonl",
    }

    files = list(
        filter(
            lambda x: sum([ll in x for ll in [langs_ner_map[lang] for lang in langs]])
            > 0,
            files,
        )
    )

    print(files)

    for file in tqdm(files):
        with open(file) as data:
            data_lines = data.read().splitlines()

        data_dicts = [json.loads(line) for line in data_lines]
        df = pd.DataFrame(data_dicts)
        df = df[~(df["target"] == "")]
        file_lang = file.split("/")[-1].split(".")[0]

        print(f"\nLanguage: {file_lang}, using file: {file}")
        print(df.head())

        df["prompt"] = df.apply(
            lambda x: x["text"] + "\n\n" + prompt_query + " ",
            axis=1,
        )
        responses = prompt_llm(model_pipeline, tokenizer, list(df["prompt"]), device)

        df["opt"] = responses
        df.to_csv(output_dir + file_lang + ".tsv", sep="\t")


def main():
    """Runs the task functions"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model_name = args.model_path
    model_pipeline, tokenizer = load_model(model_name, device)

    output_base = args.output_dir

    langs_list = [args.lang]

    sentiment(
        model_pipeline,
        tokenizer,
        f"{output_base}/sentiment/",
        device,
        langs=langs_list,
    )
    news_classification(
        model_pipeline,
        tokenizer,
        f"{output_base}/news_topic/",
        device,
        langs=langs_list,
    )
    if "amh" not in langs_list:
        cross_lingual_qa(
            model_pipeline,
            tokenizer,
            f"{output_base}/qa/",
            device,
            langs=langs_list,
            pivot=True,
        )
        cross_lingual_qa(
            model_pipeline,
            tokenizer,
            f"{output_base}/qah/",
            device,
            langs=langs_list,
            pivot=False,
        )
    machine_translation(
        model_pipeline,
        tokenizer,
        f"{output_base}/mt-to-en/",
        device,
        langs=langs_list,
        reverse=False,
    )
    machine_translation(
        model_pipeline,
        tokenizer,
        f"{output_base}/mt-from-en/",
        device,
        langs=langs_list,
        reverse=True,
    )
    named_entity_recognition(
        model_pipeline,
        tokenizer,
        f"{output_base}/ner/",
        device,
        langs=langs_list,
    )

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
