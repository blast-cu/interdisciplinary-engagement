
# ── Hyperparameters ────────────────────────────────────────────────────────────
HPARAMS = dict(
    learning_rate   = 4e-5,
    num_epochs      = 8,
    warmup_ratio    = 0.1,
    weight_decay    = 0,
    train_batch_size= 4,
    lr_scheduler     = "linear",
)

# ── Cross-validation settings ──────────────────────────────────────────────────
K_FOLDS      = 5          # number of folds
RANDOM_SEED  = 42

# Within each fold's train portion, this fraction is held out as a validation set.
VAL_FRACTION = 0.10       # e.g. 0.10 → 10 % of train becomes val

# ── Data ──────────────────────────────────────────────────────────────────────
ANNOTATED_CSV = "datasets/FINAL_MAY/annotated/citations.csv"

# Synthetic augmentation: set a path to None to skip that category.
SYNTHETIC_PATHS = {
    "substantiation + basis": "classifiers/purpose/data_splits/synthetic/synthetic_substantiation_basis.csv",
    "definition":             "classifiers/purpose/data_splits/synthetic/synthetic_definition.csv",
    "basis":                  "classifiers/purpose/data_splits/synthetic/synthetic_basis.csv",
}

SYNTHETIC_PARAPHRASED = {
    # "basis":  "classifiers/purpose/data_splits/synthetic/paraphrased/synthetic_basis_rephrased.csv",
    # "definition": "classifiers/purpose/data_splits/synthetic/paraphrased/synthetic_definition_rephrased.csv",
}
PARAPHRASED_SYNTHETIC_PATHS = [
    # 'classifiers/purpose/data_splits/synthetic/paraphrased/slice_1_25_rephrased.csv',
    # 'classifiers/purpose/data_splits/synthetic/paraphrased/slice_26_50_rephrased.csv',
    # 'classifiers/purpose/data_splits/synthetic/paraphrased/slice_51_100_rephrased.csv',
    # 'classifiers/purpose/data_splits/synthetic/paraphrased/slice_101_150_rephrased.csv',
    # 'classifiers/purpose/data_splits/synthetic/paraphrased/slice_151_200_rephrased.csv'
]

# SAMPLED_PREDICTIONS = "classifiers/purpose/data_splits/sampled_citations.csv"  # set to None to skip
SAMPLED_PREDICTIONS = None

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME    = "roberta-base"
CONTEXT_LEN   = 512
NUM_LABELS    = 7
SPAN_POOLING  = "mean"    # passed to RobertaSpanForSequenceClassification

# ── Output paths ──────────────────────────────────────────────────────────────
# Fold splits are saved here so the same splits can be reused across runs.
SPLITS_DIR    = "classifiers/purpose/k_fold/kfold_data_splits"

# Trained models + results land here.
MODELS_DIR    = "data/models"

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "interdisciplinary-citations"
WANDB_RUN_NAME = "kfold_cv"          # suffixed with fold index inside kfold_train.py

# ── Misc ──────────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 8


"""
kfold_data.py
=============
Data loading and preprocessing helpers for the k-fold pipeline.
All functions are pure (no side effects on global state) so they can be
safely called from any fold iteration.
"""

import os
import random
import unicodedata

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))  # adds project root

from classifiers.purpose.span_classifier_utils import prepare_dataset


# ── Label mapping ──────────────────────────────────────────────────────────────

LABEL_NAMES = [
    "substantiation + basis",
    "substantiation",
    "basis",
    "definition",
    "analysis",
    "use",
    "related work",
]

_LABEL_TO_INT = {name: idx for idx, name in enumerate(LABEL_NAMES)}


def map_label(txt_label: str) -> int:
    if txt_label not in _LABEL_TO_INT:
        raise ValueError(f"Unknown label: '{txt_label}'. Expected one of {list(_LABEL_TO_INT)}")
    return _LABEL_TO_INT[txt_label]


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("(", "").replace(")", "").lower()
    text = text.replace("et al.,", "et al.")
    return text


def expand_span(citation: str, context: str, n_words: int = 6) -> str:
    idx = context.find(citation)
    if idx == -1:
        return citation

    left_context = context[:idx]
    sent_end = max(left_context.rfind("."), left_context.rfind("!"), left_context.rfind("?"))
    left_words = left_context[sent_end + 1:].split()
    left_expansion = " ".join(left_words[-n_words:])
    if left_expansion:
        left_pos = context.find(left_expansion, sent_end + 1)
        left_boundary = left_pos if left_pos != -1 else idx
    else:
        left_boundary = idx

    right_boundary = idx + len(citation)
    right_context = context[right_boundary:]
    sent_end_right = min(
        (right_context.find(p) for p in [".", "!", "?"] if right_context.find(p) != -1),
        default=len(right_context),
    )
    right_words = right_context[:sent_end_right].split()
    right_expansion = " ".join(right_words[:n_words])
    if right_expansion:
        right_pos = context.find(right_expansion, right_boundary)
        right_boundary = (right_pos + len(right_expansion)) if right_pos != -1 else right_boundary

    result = context[left_boundary:right_boundary].strip()
    return result if result else citation


def is_span_findable(citation: str, context: str, tokenizer: RobertaTokenizer, max_length: int) -> bool:
    """
    Mirrors exactly what tokenize_and_map does internally.
    Returns False if the citation span would cause an IndexError during tokenization
    (e.g. because the context is too long and the span gets truncated away).
    """
    try:
        tok = tokenizer(
            context,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets     = tok["offset_mapping"][1:-1]  # strip [CLS] and [SEP]
        start_char  = context.index(citation)
        end_char    = start_char + len(citation) - 1
        token_indices = [
            i + 1
            for i, (ts, te) in enumerate(offsets)
            if (ts >= start_char and ts < end_char) or
               (te > start_char and te <= end_char) or
               (ts < start_char and te > end_char)
        ]
        return len(token_indices) > 0
    except ValueError:
        return False


# ── Span filtering ─────────────────────────────────────────────────────────────

def filter_unfindable_spans(
    records: list[dict],
    tokenizer: RobertaTokenizer,
    max_length: int,
    split_name: str = "",
    logger=None,
) -> list[dict]:
    """
    Remove records whose citation span cannot be found after tokenization.
    Called once per split in build_fold_datasets, before tokenizing.
    """
    kept    = []
    skipped = 0
    for record in records:
        if not record["citation"].strip():
            skipped += 1
            if logger:
                logger.warning(f"[{split_name}] id={record['id']}: empty citation — skipping")
            continue
        if not is_span_findable(record["citation"], record["context"], tokenizer, max_length):
            skipped += 1
            if logger:
                logger.warning(
                    f"[{split_name}] id={record['id']}: span not findable after tokenization — skipping"
                )
            continue
        kept.append(record)

    if logger and skipped:
        logger.warning(f"[{split_name}] skipped {skipped} records with unfindable spans")
    return kept


# ── Annotated data loaders ─────────────────────────────────────────────────────

def load_ann_data(data_path: str, logger=None) -> list[dict]:
    """Load a CSV of annotated examples into a list of dicts."""
    ann_df = pd.read_csv(data_path)
    ann_df = ann_df[ann_df["category"].notna()]

    rows = []
    for idx, row in ann_df.iterrows():
        clean_citation = clean_text(str(row["citation_text"]))
        clean_context  = clean_text(str(row["citation_context"]))

        if not clean_citation.strip():
            if logger:
                logger.warning(f"Row {idx}: empty citation after cleaning — skipping")
            continue
        if clean_citation not in clean_context:
            if logger:
                logger.warning(f"Row {idx}: citation not found in context — skipping")
            continue

        rows.append({
            "id":       idx,
            "citation": expand_span(clean_citation, clean_context),
            "context":  clean_context,
            "label":    map_label(row["category"]),
        })

    if logger:
        logger.info(f"Loaded {len(rows)} annotated examples from '{data_path}'")
    return rows


# ── Synthetic data loader ──────────────────────────────────────────────────────

def load_synthetic_data(category_paths: dict, logger=None) -> list[dict]:
    """
    Load synthetic CSVs for underrepresented categories.

    Parameters
    ----------
    category_paths : dict
        Keys are category label strings; values are file paths (or None to skip).

    Returns
    -------
    list of dicts with keys: id, citation, context, label
    """
    synthetic = []

    for category, path in category_paths.items():
        if path is None:
            if logger:
                logger.info(f"[synthetic] Skipping '{category}' (path is None)")
            continue
        if not os.path.exists(path):
            if logger:
                logger.warning(f"[synthetic] File not found for '{category}': {path} — skipping")
            continue

        df = pd.read_csv(path)
        required = {"citation_text", "citation_context"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Synthetic file for '{category}' is missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        label_int  = map_label(category)
        skip_count = 0
        before     = len(synthetic)

        for _, row in df.iterrows():
            clean_citation = clean_text(str(row["citation_text"]))
            clean_context  = clean_text(str(row["citation_context"]))

            if not clean_citation.strip():
                skip_count += 1
                continue
            if clean_citation not in clean_context:
                skip_count += 1
                continue

            synthetic.append({
                "id":       len(synthetic),   # re-indexed later
                "citation": expand_span(clean_citation, clean_context),
                "context":  clean_context,
                "label":    label_int,
            })

        added = len(synthetic) - before
        if logger:
            logger.warning(f"[synthetic '{category}'] skipped {skip_count} rows")
            logger.info(f"[synthetic '{category}'] loaded {added} rows from '{path}'")

    if logger:
        logger.info(f"Total synthetic examples loaded: {len(synthetic)}")
    return synthetic


# ── Sampled predictions loader ─────────────────────────────────────────────────

def load_sampled_predictions(test_df: pd.DataFrame, logger=None) -> list[dict]:
    """
    Load sampled predictions, excluding any records whose article_id
    appears in the current fold's test set.
    """
    if not SAMPLED_PREDICTIONS:
        return []

    if not os.path.exists(SAMPLED_PREDICTIONS):
        if logger:
            logger.warning(f"[sampled] File not found: {SAMPLED_PREDICTIONS} — skipping")
        return []

    test_article_ids = set(test_df["article_id"])

    df = pd.read_csv(SAMPLED_PREDICTIONS)
    df = df[df["purpose"].notna()]

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        if row["article_id"] in test_article_ids:
            skipped += 1
            continue

        clean_citation = clean_text(str(row["citation_text"]))
        clean_context  = clean_text(str(row["citation_context"]))

        if not clean_citation.strip():
            skipped += 1
            continue
        if clean_citation not in clean_context:
            skipped += 1
            continue

        rows.append({
            "id":       len(rows),
            "citation": expand_span(clean_citation, clean_context),
            "context":  clean_context,
            "label":    map_label(row["purpose"]),
        })

    if logger:
        logger.info(f"[sampled] loaded {len(rows)}, skipped {skipped} (test article_id overlap or bad citation)")
    return rows


# ── Paraphrased annotated data loader ─────────────────────────────────────────

def load_paraphrased_data(train_df: pd.DataFrame, logger=None) -> list[dict]:
    """
    Load paraphrased synthetic data, keeping only records whose
    (article_id, citation_id) pair already exists in the fold's train set.
    """
    if not PARAPHRASED_SYNTHETIC_PATHS:
        return []

    train_pairs = set(zip(train_df["article_id"], train_df["citation_id"]))

    paraphrased = []
    for path in PARAPHRASED_SYNTHETIC_PATHS:
        if not os.path.exists(path):
            if logger:
                logger.warning(f"[paraphrased] File not found: {path} — skipping")
            continue

        df = pd.read_csv(path)
        required = {"article_id", "citation_id", "citation_text", "citation_context"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Paraphrased file '{path}' is missing columns: {missing}")

        before  = len(paraphrased)
        skipped = 0

        for _, row in df.iterrows():
            pair = (row["article_id"], row["citation_id"])
            if pair not in train_pairs:
                skipped += 1
                continue

            clean_citation = clean_text(str(row["citation_text"]))
            clean_context  = clean_text(str(row["citation_context"]))

            if not clean_citation.strip():
                skipped += 1
                continue
            if clean_citation not in clean_context:
                skipped += 1
                continue

            # Look up the category from the original train row
            orig = train_df[
                (train_df["article_id"] == row["article_id"]) &
                (train_df["citation_id"] == row["citation_id"])
            ]
            if orig.empty:
                skipped += 1
                continue

            paraphrased.append({
                "id":       len(paraphrased),
                "citation": expand_span(clean_citation, clean_context),
                "context":  clean_context,
                "label":    map_label(orig.iloc[0]["category"]),
            })

        if logger:
            logger.info(f"[paraphrased] '{path}': added {len(paraphrased) - before}, skipped {skipped}")

    if logger:
        logger.info(f"Total paraphrased examples loaded: {len(paraphrased)}")
    return paraphrased


# ── Fold data assembly ─────────────────────────────────────────────────────────

def build_fold_datasets(
    fold_paths: dict,
    tokenizer: RobertaTokenizer,
    logger=None,
) -> tuple[object, object, object, torch.Tensor, DatasetDict]:
    """
    Given a fold's {train, val, test} CSV paths, return tokenized HuggingFace
    datasets and a class-weight tensor computed from the (augmented) train set.

    Returns
    -------
    (train_tokenized, val_tokenized, test_tokenized, class_weights_tensor, ds)
    """
    train_data = load_ann_data(fold_paths["train"], logger=logger)
    val_data   = load_ann_data(fold_paths["val"],   logger=logger)
    test_data  = load_ann_data(fold_paths["test"],  logger=logger)

    # ── Augment train ──────────────────────────────────────────────────────────

    def _append(train_data, new_data, label):
        if not new_data:
            return train_data
        max_id = max(d["id"] for d in train_data) if train_data else -1
        for i, entry in enumerate(new_data):
            entry["id"] = max_id + 1 + i
        train_data = train_data + new_data
        random.shuffle(train_data)
        if logger:
            logger.info(f"Train size after {label}: {len(train_data)}")
        return train_data

    synthetic             = load_synthetic_data(SYNTHETIC_PATHS, logger=logger)
    train_data            = _append(train_data, synthetic, "synthetic augmentation")

    synthetic_paraphrased = load_synthetic_data(SYNTHETIC_PARAPHRASED, logger=logger)
    train_data            = _append(train_data, synthetic_paraphrased, "synthetic paraphrased augmentation")

    train_df   = pd.read_csv(fold_paths["train"])
    train_df   = train_df[train_df["category"].notna()]
    paraphrased = load_paraphrased_data(train_df, logger=logger)
    train_data  = _append(train_data, paraphrased, "paraphrased augmentation")

    test_df    = pd.read_csv(fold_paths["test"])
    sampled    = load_sampled_predictions(test_df, logger=logger)
    train_data = _append(train_data, sampled, "sampled predictions augmentation")

    # if logger:
    #     logger.info(f"Total train size before span filtering: {len(train_data)}")

    # ── Filter unfindable spans across all splits ──────────────────────────────
    train_data = filter_unfindable_spans(train_data, tokenizer, CONTEXT_LEN, "train", logger)
    val_data   = filter_unfindable_spans(val_data,   tokenizer, CONTEXT_LEN, "val",   logger)
    test_data  = filter_unfindable_spans(test_data,  tokenizer, CONTEXT_LEN, "test",  logger)

    if logger:
        logger.info(f"Total train size after span filtering: {len(train_data)}")
    # ── Class weights from augmented, filtered train labels ────────────────────
    train_labels = [d["label"] for d in train_data]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_labels,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    # ── Tokenize ───────────────────────────────────────────────────────────────
    ds = DatasetDict({
        "train":      Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test":       Dataset.from_list(test_data),
    })

    tokenize = lambda split: prepare_dataset(
        ds[split], tokenizer,
        text_col="context", span_col="citation", label_col="label",
        max_length=CONTEXT_LEN,
    )

    return (
        tokenize("train"),
        tokenize("validation"),
        tokenize("test"),
        class_weights,
        ds,   # kept for index access inside test()
    )