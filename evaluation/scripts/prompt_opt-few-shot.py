"""
LLM few-shot evaluation script. Used for generating model responses to 
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

NUM_SHOT = 3


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
    # print(
    #     "Avg input len [toks]:",
    #     np.mean(list(map(lambda x: len(x), inputs["input_ids"]))),
    # )
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
    output = [output[i][len(messages[i]) - 2 :] for i in range(len(output))]
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

    file = list(filter(lambda x: sum([ll in x for ll in langs]) > 0, files))[0]

    df = pd.read_csv(file, sep="\t", header=0)
    df = df.sample(frac=1, random_state=42)
    language = file.split("/")[-2]
    print(f"\nLanguage: {language}, using file: {file}")
    print(df.head())

    character_cutoff = 300
    prompts = []
    labels = []
    for i in range(0, ((len(df["tweet"]) // NUM_SHOT) - 1) * NUM_SHOT, NUM_SHOT + 1):
        prompt = (
            f"Use only the following sentiment labels: positive, neutral, negative.\n\n"
        )
        for j in range(NUM_SHOT):
            prompt += f"Text: {df['tweet'].iloc[i+j][:character_cutoff]}\nSentiment: {df['label'].iloc[i+j]}\n\n"
        prompt += (
            f"Text: {df['tweet'].iloc[i + NUM_SHOT][:character_cutoff]}\nSentiment:"
        )
        prompts.append(prompt)
        labels.append(df["label"].iloc[i + NUM_SHOT])

    print(prompts[:3])
    print(labels[:3])

    print(f"Avg length of prompts", np.mean(list(map(lambda x: len(x), prompts))))

    responses = prompt_llm(model_pipeline, tokenizer, prompts, device)

    new_df = pd.DataFrame({"prompts": prompts, "label": labels, "opt": responses})
    new_df.to_csv(output_dir + language + ".tsv", sep="\t")


def news_classification(model_pipeline, tokenizer, output_dir, device, langs):
    create_dir(output_dir)
    files = glob.glob("data_repos/masakhane-news/data/**/test.tsv", recursive=True)
    assert len(files) != 0

    file = list(filter(lambda x: sum([ll in x for ll in langs]) > 0, files))[0]

    file_path = Path(file)
    df = pd.read_csv(file, sep="\t")
    df = df.sample(frac=1, random_state=42)

    label_string, _ = getlabel_string(Path(f"{file_path.parent}/labels.txt"))
    lang = file.split("/")[-2]
    print(f"\nLanguage: {lang}, using file: {file}")
    print(df.head())

    character_cutoff = 300
    prompts = []
    labels = []
    for i in range(0, ((len(df["headline"]) // NUM_SHOT) - 1) * NUM_SHOT, NUM_SHOT + 1):
        prompt = f"Use only the following topic labels: {label_string}.\n\n"
        for j in range(NUM_SHOT):
            prompt += (
                "Text: "
                + f"{df['headline'].iloc[i+j]} {df['text'].iloc[i+j]}"[
                    :character_cutoff
                ]
                + "\n"
                + f"Label: {df['category'].iloc[i+j]}\n\n"
            )
        prompt += (
            "Text: "
            + f"{df['headline'].iloc[i+j]} {df['text'].iloc[i+NUM_SHOT]}"[
                :character_cutoff
            ]
            + "\n"
            + f"Label:"
        )
        prompts.append(prompt)
        labels.append(df["category"].iloc[i + NUM_SHOT])

    print(prompts[:3])
    print(labels[:3])

    print(f"Avg length of prompts", np.mean(list(map(lambda x: len(x), prompts))))

    responses = prompt_llm(model_pipeline, tokenizer, prompts, device)

    new_df = pd.DataFrame({"prompts": prompts, "category": labels, "opt": responses})
    new_df.to_csv(output_dir + lang + ".tsv", sep="\t")


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

        character_cutoff = 700
        prompts = []
        labels = []
        answer_pivots = []
        for i in range(
            0, ((len(gp_df["context"]) // NUM_SHOT) - 1) * NUM_SHOT, NUM_SHOT + 1
        ):
            prompt = ""
            for j in range(NUM_SHOT):
                prompt += (
                    "Context: "
                    + gp_df["context"].iloc[i + j][:character_cutoff]
                    + "\n"
                    + "Question: "
                    + (
                        gp_df["question_translated"].iloc[i + j]
                        if pivot
                        else gp_df["question_lang"].iloc[i + j]
                    )
                    + "\n"
                    + f"Answer: {gp_df['answer_lang'].iloc[i+j]}\n\n"
                )
            prompt += (
                "Context: "
                + gp_df["context"].iloc[i + NUM_SHOT][:character_cutoff]
                + "\n"
                + "Question: "
                + (
                    gp_df["question_translated"].iloc[i + NUM_SHOT]
                    if pivot
                    else gp_df["question_lang"].iloc[i + NUM_SHOT]
                )
                + "\n"
                + f"Answer:"
            )
            prompts.append(prompt)
            labels.append(gp_df["question_lang"].iloc[i + NUM_SHOT])
            answer_pivots.append(gp_df["answer_pivot"].iloc[i + NUM_SHOT])

        print(prompts[:3])
        print(labels[:3])

        print(f"Avg length of prompts", np.mean(list(map(lambda x: len(x), prompts))))

        responses = prompt_llm(model_pipeline, tokenizer, prompts, device)

        new_df = pd.DataFrame(
            {
                "prompts": prompts,
                "answer_lang": labels,
                "opt": responses,
                "answer_pivot": answer_pivots,
            }
        )
        new_df.to_csv(output_dir + language + ".tsv", sep="\t")


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

        lang_mappings = {
            "yor": "Yoruba",
            "amh": "Amharic",
            "ibo": "Igbo",
            "hau": "Hausa",
        }

        target_lang = lang_mappings[target_lang_abv]
        pivot_lang = "English" if pivot_lang_abv == "en" else "French"

        print(f"\nLanguage: {target_lang}, using file: {file}")
        print(df.head())

        character_cutoff = 1000
        prompts = []
        en_labels = []
        target_lang_labels = []

        if not reverse:
            for i in range(
                0, ((len(df["en"]) // NUM_SHOT) - 1) * NUM_SHOT, NUM_SHOT + 1
            ):
                prompt = (
                    f"Translate the {pivot_lang} sentence below to {target_lang}.\n\n"
                )
                for j in range(NUM_SHOT):
                    prompt += (
                        "Sentence: "
                        + df["en"].iloc[i + j][:character_cutoff]
                        + "\n"
                        + f"Translation: {df[target_lang_abv].iloc[i+j]}\n\n"
                    )
                prompt += (
                    "Sentence: "
                    + df["en"].iloc[i + j][:character_cutoff]
                    + "\n"
                    + f"Translation:"
                )
                prompts.append(prompt)
                target_lang_labels.append(df[target_lang_abv].iloc[i + NUM_SHOT])
                en_labels.append(df["en"].iloc[i + NUM_SHOT])
        else:
            for i in range(
                0, ((len(df["en"]) // NUM_SHOT) - 1) * NUM_SHOT, NUM_SHOT + 1
            ):
                prompt = (
                    f"Translate the {target_lang} sentence below to {pivot_lang}.\n\n"
                )
                for j in range(NUM_SHOT):
                    prompt += (
                        "Sentence: "
                        + df[target_lang_abv].iloc[i + j][:character_cutoff]
                        + "\n"
                        + f"Translation: {df['en'].iloc[i+j]}\n\n"
                    )
                prompt += (
                    "Sentence: "
                    + df[target_lang_abv].iloc[i + j][:character_cutoff]
                    + "\n"
                    + f"Translation:"
                )
                prompts.append(prompt)
                target_lang_labels.append(df[target_lang_abv].iloc[i + NUM_SHOT])
                en_labels.append(df["en"].iloc[i + NUM_SHOT])

        print(prompts[:3])
        # print(labels[:3])

        print(f"Avg length of prompts", np.mean(list(map(lambda x: len(x), prompts))))

        responses = prompt_llm(model_pipeline, tokenizer, prompts, device)

        new_df = pd.DataFrame(
            {
                "prompts": prompts,
                "en": en_labels,
                f"{target_lang_abv}": target_lang_labels,
                "opt": responses,
            }
        )
        if reverse:
            new_df.to_csv(
                output_dir + f"{target_lang_abv}-{pivot_lang_abv}" + ".tsv", sep="\t"
            )
        else:
            new_df.to_csv(
                output_dir + f"{pivot_lang_abv}-{target_lang_abv}" + ".tsv", sep="\t"
            )


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

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
