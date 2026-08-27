"""
This script fine-tunes RoBERTa for source word classification using a labeled dataset. Taking a 
sentence as the input, RoBERTa encodes each (sub)token into a dense vector representation; a 
linear classification layer is then applied to the vector representation of the candidate source 
token to predict its metaphoricity on a four-point scale from 0 to 3.
"""
import os
import random
import pandas as pd
import numpy as np

import torch
from transformers import RobertaTokenizerFast, \
    RobertaConfig, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import wandb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging

from classifiers.purpose.span_classifier_utils import \
    RobertaSpanForSequenceClassification, DataCollatorForSpanClassification, prepare_dataset, test
from data_utils.utils import set_seed, save_file, get_logger, load_file

from data_utils.utils import DATA_DIR


def run_example():

    # set up environment and seed
    seed = 42
    set_seed(seed)

    model_path = "data/models"
    output_model_name = "purpose_classifier_test"
    os.makedirs(model_path, exist_ok=True)

    # when using fast tokenizer, must disable parallelism 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1) tokenizer & dataset
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)

    # toy dataset for testing
    data = {
        "train": [
            {"id": 0, "text": "Alice went to the store. She bought apples.", "span_text": "Alice went to", "label": 0},  # "Alice"
            {"id": 1, "text": "Bob likes football and pizza.", "span_text": "Bob", "label": 1},  # "Bob"
            {"id": 2, "text": "The quick brown fox jumped.", "span_text": "quick", "label": 0},  # "quick"
            {"id": 3, "text": "Mary visited Paris last summer.", "span_text": "Paris", "label": 1},  # "Paris"
        ],
        "validation": [
            {"id": 4, "text": "John arrived yesterday.", "span_text": "John", "label": 0},
            {"id": 5, "text": "They traveled to Rome.", "span_text": "Rome", "label": 1},
        ],
        "test": [
            {"id": 6, "text": "Emma loves reading science fiction novels.", "span_text": "loves", "label": 0},
            {"id": 7, "text": "They flew to Madrid for the conference.", "span_text": "Madrid for", "label": 1},
        ]

    }

    # Instead of load_dataset for inline lists, construct Dataset directly:
    train_ds = Dataset.from_list(data["train"])
    val_ds = Dataset.from_list(data["validation"])
    test_ds = Dataset.from_list(data["test"])
    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    tokenized = prepare_dataset(ds["train"], tokenizer, text_col="text", label_col="label", max_length=128)
    val_tokenized = prepare_dataset(ds["validation"], tokenizer, text_col="text", label_col="label", max_length=128)
    test_tokenized = prepare_dataset(ds["test"], tokenizer, text_col="text", label_col="label", max_length=128)
    # 2) model creation
    num_labels = 2
    config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
    model = RobertaSpanForSequenceClassification.from_pretrained(model_name, config=config, span_pooling="mean")

    # 3) collator & trainer
    collator = DataCollatorForSpanClassification(tokenizer)

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

    training_args = TrainingArguments(
        output_dir=f"{model_path}/{output_model_name}-progress",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        # logging_steps=10,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,  # important when model expects custom inputs like span_indices,
        seed=seed,
        report_to="wandb"  # Enables Weights & Biases integration
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # 4) Train!

    trainer.train()
    test(trainer, test_tokenized, ds, model_path)

    # Save final model
    trainer.save_model(f"{model_path}/{output_model_name}")

def clean_text(text):
    clean_text = text.replace("(", "").replace(")", "").lower()
    clean_text = clean_text.replace("et al.,", "et al.")
    return clean_text

def map_label(txt_label):

    if txt_label == "substantiation + basis": return 0
    elif txt_label == "substantiation": return 1
    elif txt_label == "basis": return 2
    elif txt_label == "definition": return 3
    elif txt_label == "analysis": return 4
    elif txt_label == "use": return 5
    elif txt_label == "related work": return 6
    else: 
        print(f"{txt_label} doesn't map to an accepted label")
        raise NameError()


def load_train_data(data_path, logger):
    llm_data = load_file(data_path, logger)["data"]

    def get_label(id_txt):
        if "substantiation_+_basis" in id_txt: return "substantiation + basis"
        elif "substantiation_" in id_txt: return "substantiation"
        elif "basis_" in id_txt: return "basis"
        elif "definition_" in id_txt: return "definition"
        elif "analysis_" in id_txt: return "analysis"
        elif "use_" in id_txt: return "use"
        elif "related_work_" in id_txt: return "related work"
        else: raise NameError(name="id_txt doesn't correspond to a label")

    synth_data = []
    skip_count = 0
    for idx, (sample_id, info) in enumerate(llm_data.items()):
        sample = info["sample"]
        # get label from id
        label = get_label(sample_id) # TODO: get this
        clean_citation_text = clean_text(sample["citation"])
        clean_context = clean_text(sample["context"])
        if clean_citation_text not in clean_context:
            skip_count += 1
        else:
            synth_data.append({
                "id": idx,
                "citation": clean_citation_text,
                "context": clean_context,
                "label": map_label(label)
            })

    logger.warning(f"skipped {skip_count} generated example")
    logger.info(f"Loaded {len(synth_data)} synthetic examples for training")

    return synth_data

def load_ann_data(data_path, logger):
    # load from csv
    ann_df = pd.read_csv(data_path)
    ann_df = ann_df[ann_df["category"].notna()]

    ann_set = []
    for idx, row in ann_df.iterrows():
        ann_set.append({
            "id": idx,
            "citation": clean_text(row["citation_text"]),
            "context": clean_text(row["citation_context"]),
            "label": map_label(row["category"])
        })

    logger.info(f"Loaded {len(ann_set)} annotated examples to use for testing")
    return ann_set


def main():

    seed = 42
    set_seed(seed)
    LOGGER = get_logger("train purpose classifier")
    LOGGER.info("Starting script to train citation purpose classifier")

    context_len = 512
    NUM_EPOCHS = 5

    model_path = "data/models"
    output_model_name = "purpose_classifier"
    os.makedirs(model_path, exist_ok=True)

    # when using fast tokenizer, must disable parallelism 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1) tokenizer & dataset
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)

    # load training data
    val_size = 0.1
    train_data = load_train_data(f"{DATA_DIR}dataset_gen/results/default_corpus_processed.json", logger=LOGGER)

    # split into train and val
    random.shuffle(train_data)
    n = len(train_data)
    train_split = train_data[:int((1-val_size) * n)]
    val_split = train_data[int((1-val_size) * n):]

    # load testing data
    # ann_path = f"{DATA_DIR}manual_anns/clean/citations_og.csv"
    clean_ann_path = f"{DATA_DIR}manual_anns/citations.csv"
    ann_data = load_ann_data(clean_ann_path, logger=LOGGER)

    data = {
        "train": train_split,
        "validation": val_split,
        "test": ann_data
    }

    # Instead of load_dataset for inline lists, construct Dataset directly:
    train_ds = Dataset.from_list(data["train"])
    val_ds = Dataset.from_list(data["validation"])
    test_ds = Dataset.from_list(data["test"])
    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    tokenized = prepare_dataset(ds["train"], tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    val_tokenized = prepare_dataset(ds["validation"], tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    test_tokenized = prepare_dataset(ds["test"], tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    
    num_labels = 7
    config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
    model = RobertaSpanForSequenceClassification.from_pretrained(model_name, config=config, span_pooling="mean")

    # 3) collator & trainer
    collator = DataCollatorForSpanClassification(tokenizer)

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
    
    training_args = TrainingArguments(
        output_dir=f"{model_path}/{output_model_name}-progress",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,  # important when model expects custom inputs like span_indices,
        seed=seed,
        report_to="wandb"  # Enables Weights & Biases integration
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # 4) Train!

    trainer.train()
    out_path = os.path.join(model_path, output_model_name)
    test(trainer, test_tokenized, ds, out_path)

    # Save final model
    trainer.save_model(out_path)


# If running as script:
if __name__ == "__main__":
    # run_example()
    main()