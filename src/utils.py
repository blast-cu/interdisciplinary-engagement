import random
import numpy as np
import torch

import os
import json
import logging
from dotenv import load_dotenv

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If using CUDA


def get_logger(name: str, level: int=logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to console handler
    ch.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(ch)
    
    return logger


def save_json(data: dict, json_path: str, json_name: str, indent: int=4, logger=None):
    """
    save data to a json file
    """
    out_path = os.path.join(json_path, json_name)
    try:
        with open(out_path, "w") as f:
            if indent is None:
                json.dump(data, f)
            else:
                json.dump(data, f, indent=indent)

        if logger is not None:
            logger.info(f"Saved data to: {out_path}")

    except Exception as e:
        raise IOError(f"Error saving data to {out_path}: {e}")


def load_json(file_path:str, logger=None):
    """
    Load the data from a file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if logger: logger.info(f"Loaded data from: {file_path}")

    except FileNotFoundError:
        if logger: logger.error(f"File not found: {file_path}")
        return None

    return data


def load_env(logger: logging.Logger=None):
    """
    Load the environment variables.
    """
    load_dotenv()
    vars = {
        "DB_PATH": os.environ.get("DB_PATH"),
        "OLD_DB_PATH": os.environ.get("OLD_DB_PATH"),
        "VENUE_INFO": os.environ.get("VENUE_INFO"),
        "VENUE_DATA_PATH": os.environ.get("VENUE_DATA_PATH"),
        "LABELED_PAPERS_PATH": os.environ.get("LABELED_PAPERS_PATH"),
        "CSV_DATA_PATH": os.environ.get("CSV_DATA_PATH"),
        "FINAL_PRED_PATH": os.environ.get("FINAL_PRED_PATH"),
        # "REPO_PATH": os.environ.get("REPO_PATH"),
        # "RAW_DATA_DIR": os.environ.get("RAW_DATA_DIR"),
        # "DB_DIR": os.environ.get("DB_DIR"),
        # "INTERIM_DIR": os.environ.get("INTERIM_DIR"),
        # "RESULTS_DIR": os.environ.get("RESULTS_DIR"),
        "RANDOM_SEED": int(os.environ.get("RANDOM_SEED", 42)),
        "DEVICE": os.environ.get("DEVICE", "cuda"),  # Default to 'cuda' if not set
    }

    if logger:
        logger.info("Loaded environment variables")

    return vars

def create_out_dirs(data_dir, conference, year, nlp_css_label):
    out_dir = f'{data_dir}{conference}/{year}/'
    os.makedirs(out_dir, exist_ok=True)

    raw_dir = f'{out_dir}{nlp_css_label}/raw/'
    os.makedirs(raw_dir, exist_ok=True)

    processed_dir = f'{out_dir}{nlp_css_label}/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    xml_out_dir = f"{out_dir}{nlp_css_label}/xml/"
    os.makedirs(xml_out_dir, exist_ok=True)

    dict_out_dir = f"{out_dir}{nlp_css_label}/dict/"
    os.makedirs(dict_out_dir, exist_ok=True)

    return out_dir, raw_dir, processed_dir, xml_out_dir, dict_out_dir