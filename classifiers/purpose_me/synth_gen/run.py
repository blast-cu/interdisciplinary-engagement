import json
import yaml
import os
import argparse

from data_utils.utils import DATA_DIR, load_file, save_file
from classifiers.purpose.synth_gen.ollama_utils import Annotate  # TODO: update this import statement

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

NUM_SAMPLES_PER_LABEL = 500

def format_example(ex):
    ex = "\n\n" + ex
    return ex

def generate_data(script_path):
    # read in the category definitions from the prompts directory
    prompt_dir_path = f"{script_path}/prompts/"
    label_def_path = f"{prompt_dir_path}category_defs.json"
    category_data = load_file(label_def_path, LOGGER)

    # generate entries for each category
    data_formatted = {}
    for cat, cat_info in category_data.items():
        cat_def = cat_info["definition"]
        cat_id_formatted = cat.replace(" ", "_").lower()

        # get examples
        example_0 = format_example(cat_info["examples"][0])
        example_1 = format_example(cat_info["examples"][1])

        for idx in range(NUM_SAMPLES_PER_LABEL):
            sample_id = f"{cat_id_formatted}_{str(idx)}"

            data_formatted[sample_id] = {
                "text": f"Citation Purpose: '{cat}'\n'{cat}' Definition: {cat_def}\nSample output: {example_0}{example_1}\n"
            }

    return data_formatted

"""
This script is used to generate annotations by an LLM 
for a given dataset using the Annotate class.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate a dataset using an LLM."
    )
    parser.add_argument(
        "--config", type=str, default="default.yaml",
        help="Name of config file in configs directory"
    )
    parser.add_argument(
        "--prompt_file", type=str,
        help="The name of the json file in prompts/ to be used.",
    )
    parser.add_argument(
        "--dataset", default="cat_gen.json", type=str,
        help="Name of dataset to analyze in the 'DATA_DIR/' directory.",
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        help="Name of the output file to save to in DATA_DIR/results."
    )
    # LLM arguments for ollama.
    parser.add_argument(
        "--model", type=str,
        help="ollama model id of the LLM to use for the initial exploration.",
    )
    parser.add_argument(
        "--seed", type=int, help="The seed for the random number generator."
    )
    parser.add_argument(
        "--temperature", type=float, help="The temperature for sampling."
    )
    parser.add_argument(
        '--host', metavar='HOST', required=True,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port', metavar='PORT', required=True,
        help="The port for the Ollama API."
    )

    # Arguments for parallel processing and saving.
    parser.add_argument(
        "--workers", type=int,
        help="Number of workers to use for parallel processing."
    )
    parser.add_argument(
        "--save_interval", type=int,
        help="The interval at which to save the results."
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="Max # of tries for LLM to gen correctly formatted response.",
    )

    # set up data paths
    DATA_PATH = f"{DATA_DIR}dataset_gen/"
    os.makedirs(DATA_PATH, exist_ok=True)
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    # TODO: generate samples to pass to LLM
    data = generate_data(SCRIPT_PATH)

    args = parser.parse_args()
    if args.config is not None:  # load the config file containing defaults.
        config_file = os.path.join(SCRIPT_PATH, "configs", args.config)
        config = yaml.safe_load(open(config_file)) if args.config is not None else {}
        parser.set_defaults(**config)
        args = parser.parse_args()

    # save data to DATA_PATH
    save_file(data, DATA_PATH, args.dataset, logger=LOGGER)
    # exit()
    # create the annotator object and process the documents.
    annotator = Annotate(
        args,
        SCRIPT_PATH,
        DATA_PATH,
        LOGGER
    )
    annotator.process_docs()
