"""
infer_purpose.py

Predicts citation purpose for a CSV of citations using a trained
RobertaSpanForSequenceClassification model.

Configuration is set via the constants at the top of the script.
Set THRESHOLD to None to disable confidence filtering (predict on all examples).

Output:
    Same CSV as input with an extra column `purpose`.
    Rows where max-softmax confidence < THRESHOLD get purpose = None.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaConfig
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from classifiers.purpose.span_classifier_utils import (
    RobertaSpanForSequenceClassification,
    DataCollatorForSpanClassification,
    prepare_dataset,
)
from classifiers.purpose.train_high_prec import clean_text, map_label
from src.utils import set_seed, get_logger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH   = "data/models/purpose_classifier_best_"
INPUT_FILE   = "data/datasets/all_citations.csv"
OUTPUT_FILE  = "citations_with_purpose_threshold_inference.csv"
THRESHOLD    = 0.91    # default = 0.5
# ──────────────────────────────────────────────────────────────────────────────

CONTEXT_LEN  = 512
BATCH_SIZE   = 32
SEED         = 42

LABEL_NAMES = [
    "substantiation + basis", "substantiation", "basis",
    "definition", "analysis", "use", "related work",
]

def is_span_findable(citation, context, tokenizer, max_length):
    """Mirrors exactly what tokenize_and_map does, returns False if it would crash."""
    try:
        tok = tokenizer(
            context,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = tok["offset_mapping"][1:-1]
        start_char = context.index(citation)
        end_char   = start_char + len(citation) - 1
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
        default=len(right_context),
    )
    right_words = right_context[:sent_end_right].split()
    right_expansion = ' '.join(right_words[:n_words])
    if right_expansion:
        right_pos = context.find(right_expansion, right_boundary)
        right_boundary = (right_pos + len(right_expansion)) if right_pos != -1 else right_boundary

    result = context[left_boundary:right_boundary]
    return result if result.strip() else citation


def load_input_data(csv_path, tokenizer, context_len, logger):
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    records         = []
    skipped_indices = set()

    for idx, row in df.iterrows():
        clean_citation = clean_text(row["citation_text"])
        clean_context  = clean_text(row["citation_context"])

        # Skip empty citations (malformed extraction artifacts e.g. bare years)
        if not clean_citation.strip():
            logger.warning(f"Row {idx}: empty citation after cleaning — skipping")
            skipped_indices.add(idx)
            continue

        expanded = expand_span(clean_citation, clean_context)

        if expanded not in clean_context:
            logger.warning(f"Row {idx}: citation not found in context after expansion — skipping")
            skipped_indices.add(idx)
            continue

        # tok_test = tokenizer(clean_context, truncation=True, max_length=context_len)
        # decoded  = tokenizer.decode(tok_test["input_ids"], skip_special_tokens=True)
        # if expanded not in decoded:
        #     logger.warning(f"Row {idx}: citation truncated by tokenizer — skipping")
        #     skipped_indices.add(idx)
        #     continue
        if not is_span_findable(expanded, clean_context, tokenizer, context_len):
            logger.warning(f"Row {idx}: span not findable after tokenization — skipping")
            skipped_indices.add(idx)
            continue

        records.append({
            "id":       idx,
            "citation": expanded,
            "context":  clean_context,
            "label":    0,
        })

    logger.info(f"Prepared {len(records)} examples ({len(skipped_indices)} skipped)")
    return records, skipped_indices, df


def run_inference(model, tokenized_dataset, collator, device, batch_size):
    loader    = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collator)
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")  # dummy labels, not needed
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def main():
    logger = get_logger("infer_purpose")
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Confidence threshold: {THRESHOLD if THRESHOLD is not None else 'None (predict all)'}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {MODEL_PATH}")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    config    = RobertaConfig.from_pretrained(MODEL_PATH)
    model     = RobertaSpanForSequenceClassification.from_pretrained(
        MODEL_PATH, config=config, span_pooling="mean"
    ).to(device)

    # ── Load & preprocess data ────────────────────────────────────────────────
    records, skipped_indices, original_df = load_input_data(
        INPUT_FILE, tokenizer, CONTEXT_LEN, logger
    )

    dataset   = Dataset.from_list(records)
    tokenized = prepare_dataset(
        dataset, tokenizer,
        text_col="context", span_col="citation", label_col="label",
        max_length=CONTEXT_LEN,
    )
    collator = DataCollatorForSpanClassification(tokenizer)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference...")
    probs     = run_inference(model, tokenized, collator, device, BATCH_SIZE)
    max_probs = probs.max(axis=1)
    preds     = probs.argmax(axis=1)

    # ── Apply threshold and map back to original dataframe ───────────────────
    original_df["purpose"] = None
    record_indices         = [r["id"] for r in records]

    for i, orig_idx in enumerate(record_indices):
        if THRESHOLD is None or max_probs[i] >= THRESHOLD:
            original_df.at[orig_idx, "purpose"] = LABEL_NAMES[preds[i]]

    # ── Save output ───────────────────────────────────────────────────────────
    original_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved output to {OUTPUT_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total     = len(original_df)
    n_predicted = original_df["purpose"].notna().sum()
    n_skipped   = len(skipped_indices)
    n_abstained = n_total - n_predicted - n_skipped
    logger.info(f"Total rows     : {n_total}")
    logger.info(f"Predicted      : {n_predicted}")
    logger.info(f"Below threshold: {n_abstained}")
    logger.info(f"Skipped        : {n_skipped}")


if __name__ == "__main__":
    main()