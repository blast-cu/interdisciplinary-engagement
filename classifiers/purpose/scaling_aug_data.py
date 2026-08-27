"""
scaling.py
================
Data scaling experiment: sweeps over how much sampled prediction data is added
to training, runs a single train/val/test split for each split value, and plots the results.

What "split value" means
------------------------
A split value of e.g. 1 means: "my real annotated data makes up 1% of total
training data."  Given ~N annotated samples, the number of sampled predictions
needed is:

    n_sampled = N * (100 - split_pct) / split_pct

That pool is sampled as evenly as possible across 7 classes (n_per_class =
n_sampled // 7).  If any class cannot meet n_per_class, ALL classes are capped
at the smallest available count to preserve balance.

Synthetic data (SYNTHETIC_PATHS, SYNTHETIC_PARAPHRASED) is still included on
top of this, as in the base pipeline.

W&B logging
-----------
* One W&B run per split value  →  train/val/test metrics for that split
* One summary W&B run          →  all split values in one table/chart for
                                   macro_f1, precision, recall, weighted_f1

Usage
-----
    python scaling.py
    python scaling.py --overwrite-splits  # regenerate splits
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import classifiers.purpose.utils as cfg
from classifiers.purpose.utils import (
    clean_text,
    expand_span,
    map_label,
    filter_unfindable_spans,
    LABEL_NAMES,
)
from src.utils import set_seed, get_logger

# ── Scaling config ─────────────────────────────────────────────────────────────
SPLIT_PERCENTAGES = [10, 25, 50, 75, 90, 100]

PREDICTIONS_POOL_PATH = "citations_with_purpose_threshold_inference.csv"

SUMMARY_RUN_NAME = "scaling_summary"

SUMMARY_METRICS = ["macro_f1", "precision", "recall", "weighted_f1"]

# ── Single split paths — update these to point to your pre-split CSVs ──────────
TRAIN_CSV = "classifiers/purpose/data_splits/train_data.csv"  # or hardcode: "path/to/train.csv"
VAL_CSV   = "classifiers/purpose/data_splits/val_data.csv"
TEST_CSV  = "classifiers/purpose/data_splits/test_data.csv"
# ──────────────────────────────────────────────────────────────────────────────


# ── Sampling logic ─────────────────────────────────────────────────────────────

def count_annotated_samples(annotated_csv: str) -> int:
    df = pd.read_csv(annotated_csv)
    return len(df[df["category"].notna()])


def compute_n_sampled(n_annotated: int, split_pct: float) -> int:
    total = n_annotated * 100.0 / split_pct
    return max(0, round(total - n_annotated))


def sample_predictions(
    pool_path: str,
    test_df: pd.DataFrame,
    n_sampled: int,
    seed: int,
    logger=None,
) -> list[dict]:
    if n_sampled == 0:
        return []

    df = pd.read_csv(pool_path)
    df = df[df["purpose"].notna()].reset_index(drop=True)

    test_article_ids = set(test_df["article_id"])
    before_excl = len(df)
    df = df[~df["article_id"].isin(test_article_ids)].reset_index(drop=True)
    if logger:
        logger.info(
            f"[scaling] Pool: {before_excl} total, {len(df)} after excluding "
            f"{len(test_article_ids)} test article_ids"
        )

    n_per_class = n_sampled // cfg.NUM_LABELS
    class_dfs = {}
    for label_name in LABEL_NAMES:
        class_dfs[label_name] = df[df["purpose"] == label_name]

    available_counts = {name: len(cdf) for name, cdf in class_dfs.items()}
    min_available    = min(available_counts.values())

    if min_available < n_per_class:
        if logger:
            bottleneck = min(available_counts, key=available_counts.get)
            logger.warning(
                f"[scaling] Class '{bottleneck}' only has {min_available} examples "
                f"(need {n_per_class}) — capping all classes at {min_available}"
            )
        n_per_class = min_available

    if logger:
        logger.info(
            f"[scaling] Sampling {n_per_class} per class × {cfg.NUM_LABELS} classes "
            f"= {n_per_class * cfg.NUM_LABELS} total"
        )

    records = []
    for label_name, cls_df in class_dfs.items():
        sampled_df = cls_df.sample(n=n_per_class, random_state=seed, replace=False)
        label_int  = map_label(label_name)

        for _, row in sampled_df.iterrows():
            clean_citation = clean_text(str(row["citation_text"]))
            clean_context  = clean_text(str(row["citation_context"]))

            if not clean_citation.strip():
                continue
            if clean_citation not in clean_context:
                continue

            records.append({
                "id":       len(records),
                "citation": expand_span(clean_citation, clean_context),
                "context":  clean_context,
                "label":    label_int,
            })

    if logger:
        logger.info(f"[scaling] Sampled {len(records)} records from prediction pool")
    return records


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_datasets_with_scaling(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    tokenizer,
    n_sampled: int,
    split_pct: float,
    logger=None,
):
    from classifiers.purpose.utils import (
        load_ann_data,
        load_synthetic_data,
        load_paraphrased_data,
        filter_unfindable_spans,
    )
    from datasets import Dataset, DatasetDict
    from classifiers.purpose.span_classifier_utils import prepare_dataset
    from sklearn.utils.class_weight import compute_class_weight

    train_data = load_ann_data(train_csv, logger=logger)
    val_data   = load_ann_data(val_csv,   logger=logger)
    test_data  = load_ann_data(test_csv,  logger=logger)

    def _append(base, new, label):
        if not new:
            return base
        max_id = max(d["id"] for d in base) if base else -1
        for i, entry in enumerate(new):
            entry["id"] = max_id + 1 + i
        base = base + new
        random.shuffle(base)
        if logger:
            logger.info(f"Train size after {label}: {len(base)}")
        return base

    synthetic = load_synthetic_data(cfg.SYNTHETIC_PATHS, logger=logger)
    train_data = _append(train_data, synthetic, "synthetic augmentation")

    synthetic_paraphrased = load_synthetic_data(cfg.SYNTHETIC_PARAPHRASED, logger=logger)
    train_data = _append(train_data, synthetic_paraphrased, "synthetic paraphrased augmentation")

    train_df_raw = pd.read_csv(train_csv)
    train_df_raw = train_df_raw[train_df_raw["category"].notna()]
    paraphrased  = load_paraphrased_data(train_df_raw, logger=logger)
    train_data   = _append(train_data, paraphrased, "paraphrased augmentation")

    test_df = pd.read_csv(test_csv)
    sampled = sample_predictions(
        pool_path=PREDICTIONS_POOL_PATH,
        test_df=test_df,
        n_sampled=n_sampled,
        seed=cfg.RANDOM_SEED,
        logger=logger,
    )
    train_data = _append(train_data, sampled, f"sampled predictions ({split_pct}% real data split)")

    if logger:
        logger.info(f"Total train size before span filtering: {len(train_data)}")

    train_data = filter_unfindable_spans(train_data, tokenizer, cfg.CONTEXT_LEN, "train", logger)
    val_data   = filter_unfindable_spans(val_data,   tokenizer, cfg.CONTEXT_LEN, "val",   logger)
    test_data  = filter_unfindable_spans(test_data,  tokenizer, cfg.CONTEXT_LEN, "test",  logger)

    if logger:
        logger.info(f"Total train size after span filtering: {len(train_data)}")

    train_labels = [d["label"] for d in train_data]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(cfg.NUM_LABELS),
        y=train_labels,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    ds = DatasetDict({
        "train":      Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test":       Dataset.from_list(test_data),
    })

    tokenize = lambda split: prepare_dataset(
        ds[split], tokenizer,
        text_col="context", span_col="citation", label_col="label",
        max_length=cfg.CONTEXT_LEN,
    )

    return (
        tokenize("train"),
        tokenize("validation"),
        tokenize("test"),
        class_weights,
        ds,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_scaled(
    split_pct: float,
    n_sampled: int,
    wandb_run=None,
    models_dir: str = None,
) -> dict:
    from transformers import RobertaConfig, RobertaTokenizerFast, TrainingArguments
    from classifiers.purpose.span_classifier_utils import (
        DataCollatorForSpanClassification,
        RobertaSpanForSequenceClassification,
        test as span_test,
    )
    from classifiers.purpose.train_high_prec import WeightedTrainer, compute_metrics

    hp = cfg.HPARAMS
    set_seed(cfg.RANDOM_SEED)

    logger = get_logger(f"scaling_{split_pct}pct")
    logger.info(f"=== Scaling experiment: split={split_pct}% ===")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.MODEL_NAME, add_prefix_space=True)

    train_tok, val_tok, test_tok, class_weights, ds = build_datasets_with_scaling(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        tokenizer=tokenizer,
        n_sampled=n_sampled,
        split_pct=split_pct,
        logger=logger,
    )

    roberta_cfg = RobertaConfig.from_pretrained(cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS)
    model = RobertaSpanForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, config=roberta_cfg, span_pooling=cfg.SPAN_POOLING
    )
    collator = DataCollatorForSpanClassification(tokenizer)

    run_model_dir = os.path.join(
        models_dir or cfg.MODELS_DIR,
        f"scaling_{split_pct}pct",
    )
    os.makedirs(run_model_dir, exist_ok=True)

    training_args = TrainingArguments(
        fp16=True,
        output_dir=os.path.join(run_model_dir, "progress"),
        per_device_train_batch_size=hp["train_batch_size"],
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        num_train_epochs=hp["num_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=hp["learning_rate"],
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        seed=cfg.RANDOM_SEED,
        report_to="wandb" if wandb_run else "none",
        logging_strategy="epoch",
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        lr_scheduler_type=hp["lr_scheduler"],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    val_result  = trainer.evaluate(val_tok)
    val_metrics = {k.removeprefix("eval_"): v for k, v in val_result.items()}

    out_path = os.path.join(run_model_dir, "best_model")
    span_test(trainer, test_tok, ds, out_path)

    test_result  = trainer.predict(test_tok)
    test_metrics = compute_metrics(test_result)

    trainer.save_model(out_path)

    if wandb_run:
        wandb.log({
            **{f"val/{k}":  v for k, v in val_metrics.items()},
            **{f"test/{k}": v for k, v in test_metrics.items()},
        })

    logger.info(f"Val  metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    return {
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    logger = get_logger("scaling")

    if not os.path.exists(PREDICTIONS_POOL_PATH):
        raise FileNotFoundError(
            f"Prediction pool not found: '{PREDICTIONS_POOL_PATH}'. "
            "Set PREDICTIONS_POOL_PATH at the top of scaling.py."
        )

    n_annotated = count_annotated_samples("data/ann_data/citations.csv")
    logger.info(f"Annotated sample count (with category): {n_annotated}")

    split_configs = []
    for pct in SPLIT_PERCENTAGES:
        n_sampled       = compute_n_sampled(n_annotated, pct)
        n_per_class_tgt = n_sampled // cfg.NUM_LABELS
        split_configs.append({"pct": pct, "n_sampled": n_sampled})
        logger.info(
            f"Split {pct}%: need {n_sampled} sampled predictions "
            f"({n_per_class_tgt} per class target)"
        )

    summary_run = wandb.init(
        project=cfg.WANDB_PROJECT,
        name=SUMMARY_RUN_NAME,
        config={
            **cfg.HPARAMS,
            "split_percentages": SPLIT_PERCENTAGES,
            "n_annotated":       n_annotated,
            "seed":              cfg.RANDOM_SEED,
            "model":             cfg.MODEL_NAME,
            "predictions_pool":  PREDICTIONS_POOL_PATH,
        },
        reinit=True,
    )

    all_results   = []
    summary_table = wandb.Table(
        columns=["split_pct", "n_sampled"] +
                [f"test_{m}" for m in SUMMARY_METRICS] +
                [f"val_{m}"  for m in SUMMARY_METRICS]
    )

    for sc in split_configs:
        pct       = sc["pct"]
        n_sampled = sc["n_sampled"]
        logger.info(f"\n{'═'*60}")
        logger.info(f"Starting scaling run: {pct}% real data ({n_sampled} sampled predictions)")
        logger.info(f"{'═'*60}")

        split_run = wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"scaling_{pct}pct",
            config={
                **cfg.HPARAMS,
                "split_pct":   pct,
                "n_sampled":   n_sampled,
                "n_annotated": n_annotated,
                "seed":        cfg.RANDOM_SEED,
                "model":       cfg.MODEL_NAME,
            },
            reinit=True,
        )

        result = train_scaled(
            split_pct=pct,
            n_sampled=n_sampled,
            wandb_run=split_run,
            models_dir=cfg.MODELS_DIR,
        )

        val_metrics  = result["val_metrics"]
        test_metrics = result["test_metrics"]

        split_run.finish()

        row = [pct, n_sampled]
        row += [test_metrics.get(m) for m in SUMMARY_METRICS]
        row += [val_metrics.get(m)  for m in SUMMARY_METRICS]
        summary_table.add_data(*row)

        all_results.append({
            "split_pct":    pct,
            "n_sampled":    n_sampled,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
        })

    summary_run.log({"scaling_results": summary_table})

    for sc_result in all_results:
        log_dict = {"split_pct": sc_result["split_pct"]}
        for m in SUMMARY_METRICS:
            if m in sc_result["test_metrics"]:
                log_dict[f"test_{m}"] = sc_result["test_metrics"][m]
            if m in sc_result["val_metrics"]:
                log_dict[f"val_{m}"]  = sc_result["val_metrics"][m]
        summary_run.log(log_dict)

    summary_run.finish()

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    results_path = os.path.join(cfg.MODELS_DIR, "scaling_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Full scaling results saved to '{results_path}'")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()