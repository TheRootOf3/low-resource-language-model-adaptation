"""
Script for aggregating evaluation results for further processing.

Written by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project
"Language Model Adaptation for Low-Resource African Languages".
"""

import json
import os


def load_and_combine_json(root_dir, output_file):
    combined_data = {}

    # Walk through the directory
    for dirpath, dirnames, files in os.walk(root_dir):
        for file in files:
            if file == "eval_results.json":
                full_path = os.path.join(dirpath, file)
                # Open and load the JSON file
                with open(full_path, "r") as json_file:
                    data = json.load(json_file)
                    # Use the directory name as a key in the combined dictionary
                    combined_data[list(data.keys())[0]] = list(data.values())[0]

    # Write the combined data to a new JSON file
    os.makedirs("/".join(output_file.split("/")[:-1]), exist_ok=True)
    with open(output_file, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)


def main():

    for nshot in ["zero-shot", "3-shot"]:
        for ds in ["wura", "aya"]:
            for prop in ["prop-0", "prop-0.25", "prop-0.5"]:
                for path_name in [
                    "opt_100-add",
                    "opt_2000-add",
                    "opt_100-replace",
                    "opt_2000-replace",
                ]:
                    # Specify the root directory and output file name
                    root_directory = f"./results/{nshot}/{ds}/{prop}/{path_name}"  # Change this to your root directory
                    output_json_file = (
                        f"./results_aggregated/{nshot}/{ds}/{prop}/{path_name}.json"
                    )

                    # Run the function
                    load_and_combine_json(root_directory, output_json_file)

            # Specify the root directory and output file name
            root_directory = f"./results/{nshot}/{ds}/prop-0/tokenizer-opt"  # Change this to your root directory
            output_json_file = (
                f"./results_aggregated/{nshot}/{ds}/prop-0/tokenizer-opt.json"
            )

            # Run the function
            load_and_combine_json(root_directory, output_json_file)

        # Specify the root directory and output file name
        root_directory = (
            f"./results/{nshot}/baseline"  # Change this to your root directory
        )
        output_json_file = f"./results_aggregated/{nshot}/baseline.json"

        # Run the function
        load_and_combine_json(root_directory, output_json_file)


if __name__ == "__main__":
    main()
