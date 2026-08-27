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
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # adds project root

import torch
from transformers import RobertaTokenizerFast, \
    RobertaConfig, TrainingArguments, Trainer, \
    RobertaTokenizer
from datasets import Dataset, DatasetDict
import evaluate
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import logging
from sklearn.utils.class_weight import compute_class_weight

from classifiers.purpose.span_classifier_utils import \
    RobertaSpanForSequenceClassification, DataCollatorForSpanClassification, prepare_dataset, test
from src.utils import set_seed, save_json, get_logger, load_json

RUN_RELOAD_EVAL = False

# ── Synthetic data paths ───────────────────────────────────────────────────────
# Set to None (or leave the list empty) to skip a category.
SYNTHETIC_PATHS = {
    "substantiation + basis": "classifiers/purpose/data_splits/synthetic/synthetic_substantiation_basis.csv",
    "definition":             "classifiers/purpose/data_splits/synthetic/synthetic_definition.csv",
    "basis":                  "classifiers/purpose/data_splits/synthetic/synthetic_basis.csv",
}
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("(", "").replace(")", "").lower()
    text = text.replace("et al.,", "et al.")
    return text

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


def expand_span(citation: str, context: str, n_words: int = 6) -> str:
    idx = context.find(citation)
    if idx == -1:
        return citation

    left_context = context[:idx]
    sent_end = max(left_context.rfind('.'), left_context.rfind('!'), left_context.rfind('?'))
    left_words = left_context[sent_end + 1:].split()
    left_expansion = ' '.join(left_words[-n_words:])
    if left_expansion:
        left_pos = context.find(left_expansion, sent_end + 1)
        left_boundary = left_pos if left_pos != -1 else idx
    else:
        left_boundary = idx

    right_boundary = idx + len(citation)
    right_context = context[right_boundary:]
    sent_end_right = min(
        (right_context.find(p) for p in ['.', '!', '?'] if right_context.find(p) != -1),
        default=len(right_context)
    )
    right_words = right_context[:sent_end_right].split()
    right_expansion = ' '.join(right_words[:n_words])
    if right_expansion:
        right_pos = context.find(right_expansion, right_boundary)
        right_boundary = (right_pos + len(right_expansion)) if right_pos != -1 else right_boundary

    result = context[left_boundary:right_boundary].strip()
    return result if result else citation


def load_train_data(data_path, logger):
    llm_data = load_json(data_path, logger)["data"]

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
        label = get_label(sample_id)
        clean_citation = clean_text(sample["citation"])
        clean_context  = clean_text(sample["context"])
        if clean_citation not in clean_context:
            skip_count += 1
        else:
            synth_data.append({
                "id":       idx,
                "citation": expand_span(clean_citation, clean_context),
                "context":  clean_context,
                "label":    map_label(label)
            })

    logger.warning(f"skipped {skip_count} generated example")
    logger.info(f"Loaded {len(synth_data)} synthetic examples for training")
    return synth_data

def load_ann_data(data_path, logger):
    ann_df = pd.read_csv(data_path)
    ann_df = ann_df[ann_df["category"].notna()]

    ann_set = []
    for idx, row in ann_df.iterrows():
        clean_citation = clean_text(row["citation_text"])
        clean_context  = clean_text(row["citation_context"])
        ann_set.append({
            "id":       idx,
            "citation": expand_span(clean_citation, clean_context),
            "context":  clean_context,
            "label":    map_label(row["category"])
        })

    logger.info(f"Loaded {len(ann_set)} annotated examples to use for testing")
    return ann_set


def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=1)
        
        top3_preds = np.argsort(p.predictions, axis=1)[:, ::-1][:, :3]
        top3_correct = np.array([
            true in top3
            for true, top3 in zip(p.label_ids, top3_preds)
        ])
        top3_assigned = np.array([
            true if true in top3 else top3[0]
            for true, top3 in zip(p.label_ids, top3_preds)
        ])

        label_names = [
            "substantiation + basis", "substantiation", "basis",
            "definition", "analysis", "use", "related work"
        ]

        print("\n=== Per-Class Metrics ===")
        print(classification_report(p.label_ids, predictions, target_names=label_names, zero_division=0))

        print("=== Confusion Matrix (rows=true, cols=predicted) ===")
        cm = confusion_matrix(p.label_ids, predictions)
        col_width = 14
        header = f"{'':>{col_width}}" + "".join(f"{n:>{col_width}}" for n in label_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{label_names[i]:>{col_width}}"
            for j, val in enumerate(row):
                cell = f"[{val}]" if i == j else f"{val}"
                row_str += f"{cell:>{col_width}}"
            print(row_str)
        print()
        
        return {
            "accuracy":      accuracy_score(p.label_ids, predictions),
            "top3_accuracy": float(top3_correct.mean()),
            "top3_f1":       f1_score(p.label_ids, top3_assigned, zero_division=0, average="macro"),
            "macro_f1":      f1_score(p.label_ids, predictions, zero_division=0, average="macro"),
            "weighted_f1":   f1_score(p.label_ids, predictions, zero_division=0, average="weighted"),
            "micro_f1":      f1_score(p.label_ids, predictions, zero_division=0, average="micro"),
            "precision":     precision_score(p.label_ids, predictions, zero_division=0, average="macro"),
            "recall":        recall_score(p.label_ids, predictions, zero_division=0, average="macro"),
        }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

def main():

    seed = 42
    set_seed(seed)
    LOGGER = get_logger("train purpose classifier")
    LOGGER.info("Starting script to train citation purpose classifier")

    context_len = 512
    NUM_EPOCHS = 8
    TRAIN_BATCH_SIZE = 4
    # LR = 2e-5
    LR = 4e-5
    WARMUP_RATIO = 0.05
    WEIGHT_DECAY = 0

    model_path = "data/models"
    output_model_name = "roberta_baseline"
    os.makedirs(model_path, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)

    # ── 1) Build train / val / test splits from annotated data ────────────────
    df = pd.read_csv("data/ann_data/citations.csv")
    df = df[df["category"].notna()]

    # just get train/val from citation training set
    val_size = 0.1
    train_df, val_df  = train_test_split(df, test_size=val_size, random_state=42)
    
    train_df.to_csv("classifiers/purpose/final_data_splits/train_data.csv", index=False)
    val_df.to_csv("classifiers/purpose/final_data_splits/val_data.csv",     index=False)
    # test_df.to_csv("classifiers/purpose/data_splits/test_data.csv",   index=False)

    train_data = load_ann_data('classifiers/purpose/final_data_splits/train_data.csv', logger=LOGGER)
    val_data   = load_ann_data('classifiers/purpose/finL_data_splits/val_data.csv',   logger=LOGGER)
    test_data  = load_ann_data('data/ann_data/test_citations.csv',  logger=LOGGER)

    # ── 3) Class weights (computed from the augmented train set) ──────────────
    train_labels = [d["label"] for d in train_data]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(7),
        y=train_labels
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    # ── 4) Build HuggingFace datasets ─────────────────────────────────────────
    data = {
        "train":      train_data,
        "validation": val_data,
        "test":       test_data,
    }

    train_ds = Dataset.from_list(data["train"])
    val_ds   = Dataset.from_list(data["validation"])
    test_ds  = Dataset.from_list(data["test"])
    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    tokenized      = prepare_dataset(ds["train"],      tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    val_tokenized  = prepare_dataset(ds["validation"], tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    test_tokenized = prepare_dataset(ds["test"],       tokenizer, text_col="context", span_col="citation", label_col="label", max_length=context_len)
    
    num_labels = 7
    config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
    model = RobertaSpanForSequenceClassification.from_pretrained(model_name, config=config, span_pooling="mean")

    collator = DataCollatorForSpanClassification(tokenizer)

    

    wandb.init(project="interdisciplinary-citations", name=output_model_name + 'hyperparam-tuning', config={
        "epochs":        NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size":    TRAIN_BATCH_SIZE,
        "context_len":   context_len,
        "model":         model_name,
        "class_weights": True,
        # "synthetic_data": {k: v for k, v in SYNTHETIC_PATHS.items() if v is not None},
        "seed":          seed,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_type": "linear"
    })
    
    training_args = TrainingArguments(
        fp16 = True,
        output_dir=f"{model_path}/{output_model_name}-progress",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=LR,
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        seed=seed,
        report_to="wandb",
        logging_strategy="epoch",
        warmup_ratio=0.05,
        weight_decay=0.01,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Model device: {next(model.parameters()).device}")

    trainer.train()
    out_path = os.path.join(model_path, output_model_name)
    os.makedirs(out_path, exist_ok=True)
    test(trainer, test_tokenized, ds, out_path)

    trainer.save_model(out_path)
    print(f"SAVED BEST MODEL TO: {out_path}")


    if RUN_RELOAD_EVAL:
        LOGGER.info("Reloading saved model for evaluation...")
        reloaded_model = RobertaSpanForSequenceClassification.from_pretrained(out_path, config=config)
        
        eval_trainer = Trainer(
            model=reloaded_model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics
        )
        
        results = eval_trainer.predict(test_tokenized)
        preds = np.argmax(results.predictions, axis=-1)
        
        LOGGER.info(f"Reloaded model test metrics: {results.metrics}")
        wandb.log({"reloaded_" + k: v for k, v in results.metrics.items()})
        
        print("\n=== Reloaded Model Test Results ===")
        print(f"Accuracy:    {results.metrics['test_accuracy']:.4f}")
        print(f"Macro F1:    {results.metrics['test_macro_f1']:.4f}")
        print(f"Weighted F1: {results.metrics['test_weighted_f1']:.4f}")
        print(f"Precision:   {results.metrics['test_precision']:.4f}")
        print(f"Recall:      {results.metrics['test_recall']:.4f}")


if __name__ == "__main__":
    main()