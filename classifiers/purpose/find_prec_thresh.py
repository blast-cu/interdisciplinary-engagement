"""
find_threshold.py

Sweeps the softmax confidence threshold for a trained RoBERTa citation-purpose
classifier and finds the point of **highest macro-precision while remaining
discriminative** — i.e. the model still makes predictions on a meaningful
fraction of examples (controlled by MIN_COVERAGE).

What "threshold" means here
----------------------------
After softmax, each example has a max probability p = max(softmax(logits)).
* If p >= threshold  →  predict the argmax class (the model is "confident").
* If p <  threshold  →  abstain / mark as "uncertain".

As the threshold rises:
  - Precision increases  (only high-confidence, usually correct predictions get through)
  - Coverage decreases   (fewer examples are predicted on at all)

The "discriminative" requirement
---------------------------------
A threshold so high that the model predicts on only 5% of examples is useless
in practice.  MIN_COVERAGE (default 0.50) enforces that at least 50% of the
test set must still receive a prediction.  Adjust this to taste.

Output
------
* Console table of threshold → precision / coverage / macro-F1
* Best threshold printed at the end
* PNG plot saved to  threshold_curve.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import RobertaTokenizerFast, RobertaConfig
from datasets import Dataset
import pandas as pd
from sklearn.metrics import precision_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from classifiers.purpose.span_classifier_utils import (
    RobertaSpanForSequenceClassification,
    DataCollatorForSpanClassification,
    prepare_dataset,
)
from classifiers.purpose.train_high_prec import clean_text, map_label
from src.utils import set_seed, get_logger

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH      = "data/models/purpose_classifier_best_"
TEST_CSV        = "classifiers/purpose/data_splits/test_data.csv"
CONTEXT_LEN     = 512
BATCH_SIZE      = 16
MIN_COVERAGE    = 0.50        # at least 50% of examples must receive a prediction
THRESHOLD_STEPS = 100         # number of points to sweep between 0 and 1
PLOT_PATH       = "threshold_curve.png"
SEED            = 42
# ──────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = [
    "substantiation + basis", "substantiation", "basis",
    "definition", "analysis", "use", "related work",
]


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


def load_test_data(csv_path, tokenizer, context_len, logger):
    """
    Mirrors load_ann_data from the inference script:
      - cleans text
      - expands the citation span
      - skips rows where the citation isn't found in context
      - skips rows where the citation is truncated away by the tokenizer
    The test CSV written by the training script uses columns 'citation' and
    'context' (not 'citation_text' / 'citation_context').
    """
    df = pd.read_csv(csv_path)
    df = df[df["category"].notna()]

    ann_set = []
    skip_count = 0
    for idx, row in df.iterrows():
        clean_citation = clean_text(row["citation_text"])
        clean_context  = clean_text(row["citation_context"])
        expanded       = expand_span(clean_citation, clean_context)

        if expanded not in clean_context:
            skip_count += 1
            logger.warning(f"Skipping row {idx}: citation not found in context after expansion")
            continue

        # Check citation survives tokenizer truncation
        tok_test = tokenizer(clean_context, truncation=True, max_length=context_len)
        decoded  = tokenizer.decode(tok_test["input_ids"], skip_special_tokens=True)
        if expanded not in decoded:
            skip_count += 1
            logger.warning(
                f"Skipping row {idx}: citation truncated by tokenizer "
                f"(starts at char {clean_context.find(expanded)})"
            )
            continue

        ann_set.append({
            "id":       idx,
            "citation": expanded,
            "context":  clean_context,
            "label":    map_label(row["category"]),
        })

    logger.warning(f"Skipped {skip_count} examples total")
    logger.info(f"Loaded {len(ann_set)} examples for threshold sweep")
    return ann_set


def get_logits(model, tokenized_dataset, collator, device, batch_size=16):
    """Run inference and return raw logits + true labels as numpy arrays."""
    from torch.utils.data import DataLoader

    loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collator)
    all_logits, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch  = {k: v.to(device) for k, v in batch.items()}
            out    = model(**batch)
            all_logits.append(out.logits.cpu().numpy())
            all_labels.append(labels)

    return np.concatenate(all_logits), np.concatenate(all_labels)


def sweep_thresholds(logits, labels, thresholds, min_coverage):
    """
    For each threshold, compute precision and coverage on the subset of examples
    where max-softmax >= threshold.

    Returns a list of dicts with keys: threshold, precision, coverage, macro_f1.
    """
    probs        = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    max_probs    = probs.max(axis=1)
    argmax_preds = probs.argmax(axis=1)

    results = []
    for t in thresholds:
        mask     = max_probs >= t
        coverage = mask.mean()

        if coverage < min_coverage or mask.sum() == 0:
            break  # thresholds are increasing; no point continuing

        preds_subset  = argmax_preds[mask]
        labels_subset = labels[mask]

        precision = precision_score(
            labels_subset, preds_subset,
            average="macro", zero_division=0,
        )
        macro_f1 = f1_score(
            labels_subset, preds_subset,
            average="macro", zero_division=0,
        )

        results.append({
            "threshold":   round(float(t), 4),
            "precision":   round(float(precision), 4),
            "coverage":    round(float(coverage), 4),
            "macro_f1":    round(float(macro_f1), 4),
            "n_predicted": int(mask.sum()),
        })

    return results


def find_best_threshold(results):
    """
    Among all thresholds that pass the MIN_COVERAGE floor,
    return the one with highest macro-precision.
    Ties broken by highest coverage (prefer less restrictive).
    """
    if not results:
        raise ValueError("No thresholds passed the MIN_COVERAGE constraint.")
    return max(results, key=lambda r: (r["precision"], r["coverage"]))


def plot_curves(results, best, plot_path):
    thresholds = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    coverages  = [r["coverage"]  for r in results]
    f1s        = [r["macro_f1"]  for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score", color="black")
    ax1.plot(thresholds, precisions, label="Macro Precision", color="steelblue",  lw=2)
    ax1.plot(thresholds, f1s,        label="Macro F1",        color="darkorange", lw=2, linestyle="--")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Coverage", color="grey")
    ax2.plot(thresholds, coverages, label="Coverage", color="grey", lw=1.5, linestyle=":")
    ax2.set_ylim(0, 1.05)

    ax1.axvline(best["threshold"], color="crimson", lw=1.5, linestyle="-",
                label=f"Best T={best['threshold']:.3f} (prec={best['precision']:.3f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("Precision / F1 / Coverage vs. Confidence Threshold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")

def print_class_breakdown(logits, labels, thresholds):
    """
    For each threshold in the list, print per-class true counts,
    predicted counts, and abstained counts.
    """
    probs     = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    max_probs = probs.max(axis=1)

    for threshold in thresholds:
        mask = max_probs >= threshold

        print(f"\nPer-class breakdown at threshold {threshold:.4f}")
        print(f"{'Class':<26}  {'Total':>7}  {'Predicted':>10}  {'Abstained':>10}")
        print("-" * 60)
        for cls_idx, cls_name in enumerate(LABEL_NAMES):
            cls_mask  = labels == cls_idx
            total     = cls_mask.sum()
            predicted = (cls_mask & mask).sum()
            abstained = total - predicted
            print(f"{cls_name:<26}  {total:>7}  {predicted:>10}  {abstained:>10}")
        print("-" * 60)
        total_all = len(labels)
        predicted_all = mask.sum()
        print(f"{'TOTAL':<26}  {total_all:>7}  {predicted_all:>10}  {total_all - predicted_all:>10}")


def main():
    set_seed(SEED)
    logger = get_logger("find_threshold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    # BUG FIX 3: load tokenizer from roberta-base, not MODEL_PATH
    logger.info(f"Loading model from {MODEL_PATH}")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    config    = RobertaConfig.from_pretrained(MODEL_PATH)
    model     = RobertaSpanForSequenceClassification.from_pretrained(
        MODEL_PATH, config=config, span_pooling="mean"
    ).to(device)

    # ── Load & tokenize test data ─────────────────────────────────────────────
    logger.info(f"Loading test data from {TEST_CSV}")
    # BUG FIX 1+2: use corrected load_test_data (expand_span, truncation check,
    # correct column names for the training-split CSV)
    test_data = load_test_data(TEST_CSV, tokenizer, CONTEXT_LEN, logger)
    test_ds   = Dataset.from_list(test_data)
    tokenized = prepare_dataset(
        test_ds, tokenizer,
        text_col="context", span_col="citation", label_col="label",
        max_length=CONTEXT_LEN,
    )
    collator = DataCollatorForSpanClassification(tokenizer)

    # ── Get logits ────────────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    logits, labels = get_logits(model, tokenized, collator, device, BATCH_SIZE)
    logger.info(f"Inference complete. Test set size: {len(labels)}")

    # ── Sweep thresholds ──────────────────────────────────────────────────────
    thresholds = np.linspace(0.0, 1.0, THRESHOLD_STEPS + 1)
    results    = sweep_thresholds(logits, labels, thresholds, MIN_COVERAGE)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Macro F1':>10}  {'Coverage':>10}  {'N Predicted':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['threshold']:>10.3f}  {r['precision']:>10.4f}  {r['macro_f1']:>10.4f}  "
              f"{r['coverage']:>10.4f}  {r['n_predicted']:>12}")

    # ── Best threshold ────────────────────────────────────────────────────────
    best = find_best_threshold(results)
    print("\n" + "=" * 60)
    print(f"BEST THRESHOLD (highest precision, coverage >= {MIN_COVERAGE:.0%})")
    print(f"  Threshold  : {best['threshold']:.4f}")
    print(f"  Precision  : {best['precision']:.4f}")
    print(f"  Macro F1   : {best['macro_f1']:.4f}")
    print(f"  Coverage   : {best['coverage']:.4f}  ({best['n_predicted']} / {len(labels)} examples predicted)")
    print("=" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_curves(results, best, PLOT_PATH)

    # ── Per-class breakdown at top 3 thresholds ───────────────────────────────
    top3 = sorted(results, key=lambda r: (r["precision"], r["coverage"]), reverse=True)[:3]
    top3_thresholds = [r["threshold"] for r in top3]
    print_class_breakdown(logits, labels, top3_thresholds)

    # ── Save results CSV ──────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv("threshold_sweep_results.csv", index=False)
    print("Full sweep saved to: threshold_sweep_results.csv")


if __name__ == "__main__":
    main()

# # """
# find_threshold.py

# Sweeps the softmax confidence threshold for a trained RoBERTa citation-purpose
# classifier and finds the point of **highest macro-precision across three target
# classes while remaining discriminative** — i.e. the model still makes
# predictions on a meaningful fraction of examples (controlled by MIN_COVERAGE).

# Target classes for precision optimization:
#   - substantiation + basis
#   - basis
#   - definition

# What "threshold" means here
# ----------------------------
# After softmax, each example has a max probability p = max(softmax(logits)).
# * If p >= threshold  →  predict the argmax class (the model is "confident").
# * If p <  threshold  →  abstain / mark as "uncertain".

# Output
# ------
# * Console table of threshold → precision / coverage / macro-F1
# * Best threshold printed at the end
# * PNG plot saved to  threshold_curve.png
# """

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from transformers import RobertaTokenizerFast, RobertaConfig
# from datasets import Dataset
# import pandas as pd
# from sklearn.metrics import precision_score, f1_score

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# from classifiers.purpose.span_classifier_utils import (
#     RobertaSpanForSequenceClassification,
#     DataCollatorForSpanClassification,
#     prepare_dataset,
# )
# from classifiers.purpose.train_high_prec import clean_text, map_label
# from src.utils import set_seed, get_logger

# # ── Configuration ─────────────────────────────────────────────────────────────
# MODEL_PATH      = "data/models/purpose_classifier_best_"
# # TEST_CSV        = "classifiers/purpose/data_splits/final_test_set_agreements.csv"
# TEST_CSV        = "classifiers/purpose/data_splits/test_data.csv"
# CONTEXT_LEN     = 512
# BATCH_SIZE      = 16
# MIN_COVERAGE    = 0.50
# THRESHOLD_STEPS = 100
# PLOT_PATH       = "threshold_curve.png"
# SEED            = 42
# # ──────────────────────────────────────────────────────────────────────────────

# LABEL_NAMES = [
#     "substantiation + basis", "substantiation", "basis",
#     "definition", "analysis", "use", "related work",
# ]

# TARGET_CLASSES = [
#     LABEL_NAMES.index("substantiation + basis"),
#     LABEL_NAMES.index("basis"),
#     LABEL_NAMES.index("definition"),
# ]


# def expand_span(citation: str, context: str, n_words: int = 6) -> str:
#     idx = context.find(citation)
#     if idx == -1:
#         return citation

#     left_context = context[:idx]
#     sent_end = max(left_context.rfind('.'), left_context.rfind('!'), left_context.rfind('?'))
#     left_words = left_context[sent_end + 1:].split()
#     left_expansion = ' '.join(left_words[-n_words:])
#     if left_expansion:
#         left_pos = context.find(left_expansion, sent_end + 1)
#         left_boundary = left_pos if left_pos != -1 else idx
#     else:
#         left_boundary = idx

#     right_boundary = idx + len(citation)
#     right_context = context[right_boundary:]
#     sent_end_right = min(
#         (right_context.find(p) for p in ['.', '!', '?'] if right_context.find(p) != -1),
#         default=len(right_context),
#     )
#     right_words = right_context[:sent_end_right].split()
#     right_expansion = ' '.join(right_words[:n_words])
#     if right_expansion:
#         right_pos = context.find(right_expansion, right_boundary)
#         right_boundary = (right_pos + len(right_expansion)) if right_pos != -1 else right_boundary

#     result = context[left_boundary:right_boundary]
#     return result if result.strip() else citation


# def is_span_findable(citation, context, tokenizer, max_length):
#     """Mirrors exactly what tokenize_and_map does, returns False if it would crash."""
#     try:
#         tok = tokenizer(
#             context,
#             truncation=True,
#             max_length=max_length,
#             return_offsets_mapping=True,
#         )
#         offsets = tok["offset_mapping"][1:-1]
#         start_char = context.index(citation)
#         end_char   = start_char + len(citation) - 1
#         token_indices = [
#             i + 1
#             for i, (ts, te) in enumerate(offsets)
#             if (ts >= start_char and ts < end_char) or
#                (te > start_char and te <= end_char) or
#                (ts < start_char and te > end_char)
#         ]
#         return len(token_indices) > 0
#     except ValueError:
#         return False


# def load_test_data(csv_path, tokenizer, context_len, logger):
#     df = pd.read_csv(csv_path)
#     df = df[df["category"].notna()]

#     ann_set = []
#     skip_count = 0
#     for idx, row in df.iterrows():
#         clean_citation = clean_text(row["citation_text"])
#         clean_context  = clean_text(row["citation_context"])

#         if not clean_citation.strip():
#             skip_count += 1
#             logger.warning(f"Row {idx}: empty citation after cleaning — skipping")
#             continue

#         expanded = expand_span(clean_citation, clean_context)

#         if expanded not in clean_context:
#             skip_count += 1
#             logger.warning(f"Skipping row {idx}: citation not found in context after expansion")
#             continue

#         if not is_span_findable(expanded, clean_context, tokenizer, context_len):
#             skip_count += 1
#             logger.warning(f"Row {idx}: span not findable after tokenization — skipping")
#             continue

#         ann_set.append({
#             "id":       idx,
#             "citation": expanded,
#             "context":  clean_context,
#             "label":    map_label(row["category"]),
#         })

#     logger.warning(f"Skipped {skip_count} examples total")
#     logger.info(f"Loaded {len(ann_set)} examples for threshold sweep")
#     return ann_set


# def get_logits(model, tokenized_dataset, collator, device, batch_size=16):
#     """Run inference and return raw logits + true labels as numpy arrays."""
#     from torch.utils.data import DataLoader

#     loader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collator)
#     all_logits, all_labels = [], []

#     model.eval()
#     with torch.no_grad():
#         for batch in loader:
#             labels = batch.pop("labels").numpy()
#             batch  = {k: v.to(device) for k, v in batch.items()}
#             out    = model(**batch)
#             all_logits.append(out.logits.cpu().numpy())
#             all_labels.append(labels)

#     return np.concatenate(all_logits), np.concatenate(all_labels)


# def sweep_thresholds(logits, labels, thresholds, min_coverage):
#     probs        = torch.softmax(torch.tensor(logits), dim=-1).numpy()
#     max_probs    = probs.max(axis=1)
#     argmax_preds = probs.argmax(axis=1)

#     results = []
#     for t in thresholds:
#         mask     = max_probs >= t
#         coverage = mask.mean()

#         if coverage < min_coverage or mask.sum() == 0:
#             break

#         preds_subset  = argmax_preds[mask]
#         labels_subset = labels[mask]

#         precision = precision_score(
#             labels_subset, preds_subset,
#             labels=TARGET_CLASSES,
#             average="macro", zero_division=0,
#         )
#         macro_f1 = f1_score(
#             labels_subset, preds_subset,
#             labels=TARGET_CLASSES,
#             average="macro", zero_division=0,
#         )

#         results.append({
#             "threshold":   round(float(t), 4),
#             "precision":   round(float(precision), 4),
#             "coverage":    round(float(coverage), 4),
#             "macro_f1":    round(float(macro_f1), 4),
#             "n_predicted": int(mask.sum()),
#         })

#     return results


# def find_best_threshold(results):
#     if not results:
#         raise ValueError("No thresholds passed the MIN_COVERAGE constraint.")
#     return max(results, key=lambda r: (r["precision"], r["coverage"]))


# def print_class_breakdown(logits, labels, thresholds):
#     probs     = torch.softmax(torch.tensor(logits), dim=-1).numpy()
#     max_probs = probs.max(axis=1)

#     for threshold in thresholds:
#         mask = max_probs >= threshold

#         print(f"\nPer-class breakdown at threshold {threshold:.4f}")
#         print(f"{'Class':<26}  {'Total':>7}  {'Predicted':>10}  {'Abstained':>10}")
#         print("-" * 60)
#         for cls_idx, cls_name in enumerate(LABEL_NAMES):
#             cls_mask  = labels == cls_idx
#             total     = cls_mask.sum()
#             predicted = (cls_mask & mask).sum()
#             abstained = total - predicted
#             marker    = " *" if cls_idx in TARGET_CLASSES else ""
#             print(f"{cls_name:<26}{marker:<2}  {total:>7}  {predicted:>10}  {abstained:>10}")
#         print("-" * 60)
#         total_all     = len(labels)
#         predicted_all = mask.sum()
#         print(f"{'TOTAL':<28}  {total_all:>7}  {predicted_all:>10}  {total_all - predicted_all:>10}")
#     print("\n* = target class used for precision optimization")


# def plot_curves(results, best, plot_path):
#     thresholds = [r["threshold"] for r in results]
#     precisions = [r["precision"] for r in results]
#     coverages  = [r["coverage"]  for r in results]
#     f1s        = [r["macro_f1"]  for r in results]

#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.set_xlabel("Confidence Threshold")
#     ax1.set_ylabel("Score", color="black")
#     ax1.plot(thresholds, precisions, label="Macro Precision (target classes)", color="steelblue",  lw=2)
#     ax1.plot(thresholds, f1s,        label="Macro F1 (target classes)",        color="darkorange", lw=2, linestyle="--")
#     ax1.set_ylim(0, 1.05)

#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Coverage", color="grey")
#     ax2.plot(thresholds, coverages, label="Coverage", color="grey", lw=1.5, linestyle=":")
#     ax2.set_ylim(0, 1.05)

#     ax1.axvline(best["threshold"], color="crimson", lw=1.5, linestyle="-",
#                 label=f"Best T={best['threshold']:.3f} (prec={best['precision']:.3f})")

#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

#     plt.title("Precision / F1 / Coverage vs. Confidence Threshold\n"
#               "(precision & F1 over: substantiation+basis, basis, definition)")
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=150)
#     print(f"\nPlot saved to: {plot_path}")


# def main():
#     set_seed(SEED)
#     logger = get_logger("find_threshold")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # load model etc
#     logger.info(f"Loading model from {MODEL_PATH}")
#     tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
#     config    = RobertaConfig.from_pretrained(MODEL_PATH)
#     model     = RobertaSpanForSequenceClassification.from_pretrained(
#         MODEL_PATH, config=config, span_pooling="mean"
#     ).to(device)


#     logger.info(f"Loading test data from {TEST_CSV}")
#     test_data = load_test_data(TEST_CSV, tokenizer, CONTEXT_LEN, logger)
#     test_ds   = Dataset.from_list(test_data)
#     tokenized = prepare_dataset(
#         test_ds, tokenizer,
#         text_col="context", span_col="citation", label_col="label",
#         max_length=CONTEXT_LEN,
#     )
#     collator = DataCollatorForSpanClassification(tokenizer)

#     logger.info("Running inference on test set...")
#     logits, labels = get_logits(model, tokenized, collator, device, BATCH_SIZE)
#     logger.info(f"Inference complete. Test set size: {len(labels)}")

#     thresholds = np.linspace(0.0, 1.0, THRESHOLD_STEPS + 1)
#     results    = sweep_thresholds(logits, labels, thresholds, MIN_COVERAGE)

#     print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Macro F1':>10}  {'Coverage':>10}  {'N Predicted':>12}")
#     print("(precision & F1 over target classes: substantiation+basis, basis, definition)")
#     print("-" * 60)
#     for r in results:
#         print(f"{r['threshold']:>10.3f}  {r['precision']:>10.4f}  {r['macro_f1']:>10.4f}  "
#               f"{r['coverage']:>10.4f}  {r['n_predicted']:>12}")

#     best = find_best_threshold(results)
#     print("\n" + "=" * 60)
#     print(f"BEST THRESHOLD (highest precision on target classes, coverage >= {MIN_COVERAGE:.0%})")
#     print(f"  Threshold  : {best['threshold']:.4f}")
#     print(f"  Precision  : {best['precision']:.4f}")
#     print(f"  Macro F1   : {best['macro_f1']:.4f}")
#     print(f"  Coverage   : {best['coverage']:.4f}  ({best['n_predicted']} / {len(labels)} examples predicted)")
#     print("=" * 60)

#     top3 = sorted(results, key=lambda r: (r["precision"], r["coverage"]), reverse=True)[:3]
#     top3_thresholds = [r["threshold"] for r in top3]
#     print_class_breakdown(logits, labels, top3_thresholds)

#     plot_curves(results, best, PLOT_PATH)

#     pd.DataFrame(results).to_csv("threshold_sweep_results.csv", index=False)
#     print("Full sweep saved to: threshold_sweep_results.csv")


# if __name__ == "__main__":
#     main()