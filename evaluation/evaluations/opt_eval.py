"""
Calculating standardized metrics for generated model responses. 

Adapted and optimized from https://github.com/JessicaOjo/LLM-African-eval
by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 

"""

import glob
import json
import argparse

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics import f1_score
import pandas as pd

import utils_opt


def senti_eval(prediction_files):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        nans_ratio = df["opt"].isna().sum() / len(df["opt"])
        print(f"NaNs ratio: {nans_ratio}")

        df["opt_split"] = df["opt"].astype(str).str.lower()
        df["opt_split"] = df["opt_split"].str.replace("\n", " ", regex=True)
        df["opt_split"] = df["opt_split"].str.replace(r"([^:\w\s{{}}])", "", regex=True)
        df.fillna("unknown", inplace=True)
        df["opt_split"] = df["opt_split"].apply(utils_opt.normalize_senti_text)

        lang = file.split("/")[-1].split(".")[0]
        df["opt_label"] = df.apply(
            utils_opt.opt_extract_senti_label, axis=1, args=(lang,)
        )

        print(f"Example sentiment label: {str(df['opt_label'][0])}")

        df = utils_opt.filter_senti_labels(df)

        f1 = round(f1_score(df["label"], df["opt_label"], average="weighted") * 100, 2)

    return {"f1": f1, "nan_ratio": nans_ratio}


def ner_eval(prediction_files):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t", header=0)
        nans_ratio = df["opt"].isna().sum() / len(df["opt"])
        print(f"NaNs ratio: {nans_ratio}")

        df["opt"] = df["opt"].astype(str).str.lower()
        df["opt"] = df["opt"].str.replace("\n", " ", regex=True)
        df["opt"] = df["opt"].str.replace("</s>", "", regex=True)
        df["opt"] = df["opt"].str.split("please").str[0].str.strip()
        df["opt"] = df["opt"].str.split("i hope").str[0].str.strip()

        df["opt"] = df.apply(utils_opt.opt_extract_ner_pred, axis=1)

        df["target"] = df["target"].apply(utils_opt.format_ner_text, target=True)
        df["opt"] = df["opt"].apply(utils_opt.format_ner_text, target=False)
        df = df[~(df.target == "")]

        print(f"Example ner generations: {df['opt'].head()}")

        f1 = utils_opt.calculate_ner_metrics(df, "opt")

    return {"f1": f1, "nan_ratio": nans_ratio}


def mt_eval(prediction_files):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t", engine="python")
        nans_ratio = df["opt"].isna().sum() / len(df["opt"])
        print(f"NaNs ratio: {nans_ratio}")

        df["opt"] = df["opt"].astype(str).str.lower()
        df["opt_split"] = df["opt"].str.replace("\n\n", " ", regex=False)
        df["opt_split"] = df["opt_split"].str.replace("\n", " ", regex=False)
        df["opt_split"] = df["opt_split"].str.replace("</s>", " ", regex=False)
        df["opt_split"] = (
            df["opt_split"].str.split("with that said").str[-1].str.strip()
        )
        df["opt_split"] = (
            df["opt_split"]
            .str.split("with those limitations in mind")
            .str[-1]
            .str.strip()
        )
        df["opt_split"] = (
            df["opt_split"]
            .str.split("with those considerations in mind")
            .str[-1]
            .str.strip()
        )

        lang_full = file.split("/")[-1].split(".")[0]

        if lang_full.split("-")[1] == "eng":
            lang = "eng_Latn"
            language = "english"
        elif lang_full.split("-")[1] == "fra":
            lang = "fra_Latn"
            language = "french"
        elif lang_full.split("-")[1] == "deu":
            lang = "deu_Latn"
            language = "german"
        else:
            lang = lang_full.split("-")[1]
            language = utils_opt.lang_dict[lang].lower()

        df["opt_reponse"] = df.apply(
            utils_opt.opt_extract_mt_pred, axis=1, args=(language,)
        )
        df["opt_reponse"] = (
            df["opt_reponse"].str.split("i hope this helps").str[0].str.strip()
        )
        df["opt_reponse"] = (
            df["opt_reponse"].str.split("i hope that helps").str[0].str.strip()
        )
        df["opt_reponse"] = (
            df["opt_reponse"].str.split("please note that").str[0].str.strip()
        )

        df[[lang, "opt_reponse"]] = df[[lang, "opt_reponse"]].map(
            utils_opt.normalize_text
        )

        print(f"Example mt response: {str(df['opt_reponse'][0])}")

        lang_metric = utils_opt.calculate_mt_metrics(df, "opt_reponse", lang)
        lang_metric["nan_ratio"] = nans_ratio

    return lang_metric


def qa_eval(prediction_files):
    lang_metric = {}
    for file in prediction_files:

        df = pd.read_csv(file, sep="\t", engine="python")
        nans_ratio = df["opt"].isna().sum() / len(df["opt"])
        print(f"NaNs ratio: {nans_ratio}")

        df["translated_answer"] = df["answer_pivot"].apply(
            lambda x: (
                x.split(": ")[-1].strip("['").rstrip("']}")
                if not isinstance(x, float) and x is not None
                else ""
            )
        )

        df["opt_response"] = df["opt"].astype(str).str.lower()
        # df["opt_response"] = (
        #     df["opt_response"].str.split("information provided,").str[-1].str.strip()
        # )
        # df["opt_response"] = (
        #     df["opt_response"].str.split("information provided").str[-1].str.strip()
        # )
        df["opt_response"] = (
            df["opt_response"].str.split("answer :").str[-1].str.strip()
        )
        # df["opt_response"] = df["opt_response"].str.split("\n").str[-1].str.strip()
        df["opt_response"] = df["opt_response"].str.replace("\n", " ", regex=True)
        df["opt_response"] = df["opt_response"].str.replace("</s>", "", regex=False)

        df[["opt_response", "translated_answer"]] = df[
            ["opt_response", "translated_answer"]
        ].map(utils_opt.normalize_text)
        df = df[~(df["translated_answer"] == "")]

        df["opt_response"] = df.apply(utils_opt.check_yes_no, axis=1)
        print(f"Example qa generations: {str(df['opt_response'][0])}")

        lang_metric = utils_opt.calculate_qa_metrics(df, "opt_response")
        lang_metric["nan_ratio"] = nans_ratio

    return lang_metric


def news_eval(prediction_files):
    for file in prediction_files:
        df = pd.read_csv(file, sep="\t")
        nans_ratio = df["opt"].isna().sum() / len(df["opt"])
        print(f"NaNs ratio: {nans_ratio}")

        df["opt"] = df["opt"].astype(str).str.lower()
        df["opt_split"] = df["opt"].str.replace("\n\n", " ", regex=False)
        df["opt_split"] = df["opt_split"].str.replace("\n", " ", regex=False)
        df["opt_split"] = df["opt_split"].str.replace("</s>", "", regex=False)
        df["opt_split"].fillna("", inplace=True)

        df["opt_label"] = df.apply(utils_opt.opt_extract_news_label, axis=1)
        df[["category", "opt_label"]] = df[["category", "opt_label"]].map(
            utils_opt.normalize_text
        )

        # if it contains more than one label
        df["opt_label"] = df["opt_label"].apply(
            lambda x: "unknown" if x.count(" ") >= 1 else x
        )

        df["opt_label"] = df["opt_label"].apply(lambda x: utils_opt.verbalizer(x))

        # assign random labels to unknowns
        df["opt_label"] = df.apply(utils_opt.assign_label, axis=1, row_name="opt_label")

        print(
            f"Example news label: {str(df['opt_label'][0])} True label: {str(df['category'][0])}"
        )

        f1 = round(
            (f1_score(df["category"], df["opt_label"], average="weighted") * 100), 2
        )

    return {"f1": f1, "nan_ratio": nans_ratio}


def eval_task(generations_dir, task_function):
    prediction_files = glob.glob(f"{generations_dir}/*tsv", recursive=True)

    return task_function(prediction_files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generations_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--no_ner", action="store_true")

    args, _ = parser.parse_known_args()

    combined_results = {}

    print("--- QA Eval ---")
    combined_results["qa"] = eval_task(f"{args.generations_directory}/qa", qa_eval)

    print("--- QAh Eval ---")
    combined_results["qah"] = eval_task(f"{args.generations_directory}/qah", qa_eval)

    print("--- MT from en Eval ---")
    combined_results["mt-from-en"] = eval_task(
        f"{args.generations_directory}/mt-from-en", mt_eval
    )

    print("--- MT to en Eval ---")
    combined_results["mt-to-en"] = eval_task(
        f"{args.generations_directory}/mt-to-en", mt_eval
    )
    if not args.no_ner:
        print("--- NER Eval ---")
        combined_results["ner"] = eval_task(
            f"{args.generations_directory}/ner", ner_eval
        )

    print("--- News topic Eval ---")
    combined_results["news_topic"] = eval_task(
        f"{args.generations_directory}/news_topic", news_eval
    )

    print("--- Sentiment Eval ---")
    combined_results["sentiment"] = eval_task(
        f"{args.generations_directory}/sentiment", senti_eval
    )

    utils_opt.create_dir(args.output_directory)
    with open(f"{args.output_directory}/eval_results.json", "w") as outfile:
        json.dump({args.lang: combined_results}, outfile)


if __name__ == "__main__":
    main()
