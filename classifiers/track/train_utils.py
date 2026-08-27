import pandas as pd
import numpy as np
from sklearn.utils import compute_class_weight
import torch
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, Trainer
import wandb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.utils import set_seed, save_to_json
import sqlite3
import torch.nn as nn
import logging

from data_utils.utils import DB_FILENAME

def format_abstract(title: str, abstract: str) -> str:
    """
    Format the abstract by combining title and abstract text.
    """
    return title + "\n " + abstract

def get_labelled_data(logger: logging.Logger) -> list:
    """
    Load labelled data from the database.
    Each data point is a dict with keys: 'id', 'text', 'label'
    """
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    query = "SELECT id, title, abstract, nlp_css FROM article WHERE nlp_css IS NOT NULL AND abstract is NOT NULL"  # Adjust table and column names as needed
    cur.execute(query)
    rows = cur.fetchall()
    labelled_data = [{"id": row[0], "text": format_abstract(row[1], row[2]), "label": row[3]} for row in rows]
    logger.info(f"Loaded {len(labelled_data)} labelled data points from database.")

    query = "SELECT id, title, abstract, nlp_css FROM article WHERE nlp_css IS NOT NULL AND abstract is NULL"  # Adjust table and column names as needed
    cur.execute(query)
    rows = cur.fetchall()
    logger.warning(f"{len(rows)} articles missing abstracts")
    con.close()

    return labelled_data

def test(trainer: Trainer, test_tokenized: Dataset, ds: DatasetDict, model_path: str, logger: logging.Logger) -> None:
    """
    Test the trained model and log results.
    """
    test_results = trainer.predict(test_tokenized)
    print("Test metrics:", test_results.metrics)
    wandb.log(test_results.metrics)

    preds = np.argmax(test_results.predictions, axis=-1)
    preds = preds.tolist()
    labels = test_results.label_ids.tolist()
    print("Test predictions:", preds)
    print("Test labels:     ", labels)

    out_test_results = {
        "results": test_results.metrics,
        "train": pd.DataFrame(ds["train"].to_dict()).to_dict(orient='records'),
        "validation": pd.DataFrame(ds["validation"].to_dict()).to_dict(orient='records'),
        "test": pd.DataFrame(ds["test"].to_dict()).to_dict(orient='records')
    }
    for idx, _ in enumerate(ds["test"]):
        out_test_results["test"][idx]["prediction"] = preds[idx]

    save_to_json(out_test_results, model_path, "test_results.json")


def prepare_datasets(datasets: DatasetDict, tokenizer: RobertaTokenizerFast, text_col="text", label_col="label", max_length=512) -> dict:
    """
    Tokenize and map datasets for classification.
    """

    def tokenize_and_map(example):
        tok = tokenizer(
            example[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        if label_col in example:
            tok["labels"] = example[label_col]

        tok["ids"] = example["id"]

        return tok

    tokenized_datasets = {
        split: datasets[split].map(tokenize_and_map, remove_columns=datasets[split].column_names)
        for split in datasets.keys()
    }
    return tokenized_datasets

class DataCollatorForClassification:
    """
    Pads a list of examples (dicts) using tokenizer.pad, and collates span indices as tensor.
    Expects each example to contain:
      - input_ids, attention_mask
      - labels (int)
      - span_token_indices (list/tuple of two ints)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        # tokenizer.pad will pad input_ids, attention_mask, etc.
        labels = [f["labels"] for f in features] if "labels" in features[0] else None
        # Remove keys we will not pass to tokenizer.pad
        to_pad = [{k: v for k, v in f.items() if k not in ["ids", "labels"]} for f in features]
        batch = self.tokenizer.pad(to_pad, return_tensors="pt")
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch
    

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, predictions),
        "macro_f1": f1_score(p.label_ids, predictions, zero_division=0, average="macro"), # 'weighted', 'macro', 'micro'
        "weighted_f1": f1_score(p.label_ids, predictions, zero_division=0, average="weighted"), # 'weighted', 'macro', 'micro'
        "micro_f1": f1_score(p.label_ids, predictions, zero_division=0, average="micro"), # 'weighted', 'macro', 'micro'
        "precision": precision_score(p.label_ids, predictions, zero_division=0, average="macro"),
        "recall": recall_score(p.label_ids, predictions, zero_division=0, average="macro"),
    }