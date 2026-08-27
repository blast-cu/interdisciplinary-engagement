import argparse
import os
from tqdm import tqdm
import logging
from pydantic import BaseModel
import csv


from src.utils import load_env, get_logger, load_json
from classifiers.purpose_me.ollama_utils import Annotator, LLMExperimentConfig
valid_labels = ["substantiation + basis", "substantiation", "basis", "definition", "analysis", "use", "related work"]

class PurposeAnswer(BaseModel):
    category: str

def format_docs(data):
    formatted_docs = {}
    for entry in data:
        formatted_docs[str(entry["idx"])] = {
            "text": f"Citation text: {entry["citation_text"]}\nContext: {entry["citation_context"]}"
        }

    return formatted_docs


def load_data(input_path: str, valid_labels: set[str], logger=None) -> tuple[list[dict], list[dict]]:
    """
    Loads CSV with columns: citation_text, citation_context, category.

    Validates each row and separates clean rows from invalid ones.
    Invalid rows (empty fields, unrecognised label) are excluded from inference
    and written to the skipped file at the end — they never enter y_true/y_pred.

    Returns:
        examples: list of clean dicts ready for inference
        invalid:  list of dicts describing rows that failed validation
    """
    examples = []
    invalid  = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            citation_text    = row.get("citation_text",    "").strip()
            citation_context = row.get("citation_context", "").strip()
            label            = row.get("category",         "").strip()

            problems = []
            if not citation_text:
                problems.append("empty citation_text")
            if not citation_context:
                problems.append("empty citation_context")
            if not label:
                problems.append("empty category")
            elif label not in valid_labels:
                problems.append(f"unrecognised category: '{label}'")

            if problems:
                invalid.append({
                    "csv_row":        i,
                    "citation_text":  citation_text,
                    "label":          label,
                    "reasons":        problems,
                })
                if logger: logger.warning(f"Row {i} skipped at load time: {problems}")
                continue

            examples.append({
                "idx":              len(examples),   # compact sequential index into examples list
                "csv_row":          i,               # original row number for traceability
                "citation_text":    citation_text,
                "citation_context": citation_context,
                "label":            label,
            })

    if logger: logger.info(
        f"Loaded {len(examples)} valid examples from {input_path} "
        f"({len(invalid)} rows skipped at load time)."
    )
    if invalid:
        if logger: logger.warning(
            f"Skipped {len(invalid)} rows during loading — "
            f"see 'invalid_rows' in the output JSON for details."
        )
    return examples, invalid



def main(args, env_vars, logger):
    
    # for all the json files in results/metaphor_classification/llm
    # read in the file and get the verb
    REPO_DIR="/projects/alle5715/cross_disciplinary_engagement/"
    TASK_DIR="/projects/alle5715/cross_disciplinary_engagement/results"

    # INPUT_DATA_PATH = "data/ann_data/test_citations.csv"
    INPUT_DATA_PATH = "data/ann_data/all_citations.csv"


    examples, invalid = load_data(input_path=INPUT_DATA_PATH, valid_labels=valid_labels)
    logger.info(f"Loaded {len(examples)} valid entries, {len(invalid)} invalid")

    ## FORMAT TEXT FOR PROMPTING
    data_formatted = format_docs(examples)

    ## SET UP EXPERIMENT CONFIGURATION
    config = LLMExperimentConfig(
        logger=logger,
        random_seed=42,
        prompt_file=args.prompt_file,
        out_file=args.out_file,
        # model="llama3.2:3b",
        model="qwen3",
        temperature=1,
        host=args.host,
        port=args.port,
        answer_format=PurposeAnswer,
        repo_dir=REPO_DIR,
        task_dir=TASK_DIR,
        workers=8,
        max_retries=5
    )

    ### SET UP LLM ANNOTATOR
    annotator = Annotator(
        config=config,
        data_entries=data_formatted
    )
    annotator.process_docs()       
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a trained metaphor model to measure the metaphoricity of a dataset.")

    parser.add_argument(
        "--prompt_file", type=str,
        help="The name of the json file in src/llm_ann/prompts/ to be used.",
    )
    parser.add_argument( 
        "--out_file", type=str, default=None,
        help="The desired name of the output file."
    )
    parser.add_argument(
        '--host', metavar='HOST', required=True,
        help="The host for the Ollama API."
    )
    parser.add_argument(
        '--port', metavar='PORT',
        help="The port for the Ollama API."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="The number of examples to add to the prompt. By default, just uses all examples included in the prompt json. "
    )

    ### SET UP ENVIRONMENT VARIABLES AND LOGGER
    logger = get_logger(f"process_source_verbs", level=logging.INFO)
    env_vars = load_env(logger=logger)

    args = parser.parse_args()
    main(args, env_vars, logger)