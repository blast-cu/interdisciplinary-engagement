"""
Ollama-based citation purpose classification.

Parallelism: asyncio.gather sends all requests concurrently to a single Ollama
server, which manages GPU utilisation internally. This matches the pattern in
your working CURC scripts.

Usage:
  1. Set INPUT_DATA_PATH, OUTPUT_PATH, PROMPT_FILE, MODEL_NAME below.
  2. Submit via run_inference.sh (which starts ollama serve before calling this).
"""

import asyncio
import csv
import json
import logging
import random
import re
import time
from pathlib import Path

from ollama import AsyncClient
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DATA_PATH = "/projects/alle5715/cross_disciplinary_engagement/classifiers/purpose/citations.csv"    # citation_text, citation_context, label
OUTPUT_PATH     = "/projects/alle5715/cross_disciplinary_engagement/results" # directory; created if absent
PROMPT_FILE     = "/projects/alle5715/cross_disciplinary_engagement/classifiers/purpose/prompts.py"

MODEL_NAME      = "qwen3"   # swap to "qwen3:14b" for the second run

# How many requests to fire concurrently.
# Ollama queues extras internally. 8–16 is a good starting point for A100s;
# raise if GPUs are underutilised, lower if you see timeout errors.
CONCURRENCY     = 8

MAX_RETRIES     = 3     # per example, if response cannot be parsed
TEMPERATURE     = 1.0   # greedy decoding — essential for reproducible classification
MAX_TOKENS      = 64    # labels are short; cap prevents runaway generation
RESPONSE_SAMPLE = 10    # how many raw responses to print after inference

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_prompt_config(prompt_file: str):
    """
    Imports JUDGE_PROMPT, CATEGORY_DESCRIPTIONS, CATEGORY_EXAMPLES from prompts.py.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompts", prompt_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    judge_prompt = getattr(mod, "JUDGE_PROMPT", None)
    descriptions = getattr(mod, "CATEGORY_DESCRIPTIONS", {})
    examples     = getattr(mod, "CATEGORY_EXAMPLES", {})

    if judge_prompt is None:
        raise ValueError(f"JUDGE_PROMPT not found in {prompt_file}")
    if not descriptions:
        raise ValueError(f"CATEGORY_DESCRIPTIONS is empty or missing in {prompt_file}")

    valid_labels = set(descriptions.keys())
    logger.info(
        f"Loaded {len(valid_labels)} categories and "
        f"{sum(len(v) for v in examples.values())} examples from {prompt_file}."
    )
    return judge_prompt, descriptions, examples, valid_labels


def build_options_block(descriptions: dict, examples: dict) -> str:
    """Builds the {options} block interpolated into JUDGE_PROMPT."""
    parts = []
    for label, description in descriptions.items():
        block = f"{label}\nCriteria:\n{description.strip()}"
        category_examples = examples.get(label, [])
        if category_examples:
            ex_lines = []
            for i, ex in enumerate(category_examples, 1):
                ex_lines.append(
                    f"  Example {i}:\n"
                    f"    Citation context: {ex['citation_context'].strip()}\n"
                    f"    Citation text: {ex['citation_text'].strip()}"
                )
            block += "\nExamples:\n" + "\n".join(ex_lines)
        parts.append(block)
    return "\n\n".join(parts)


def build_prompt(citation_text: str, citation_context: str, judge_prompt: str, options_block: str) -> str:
    return judge_prompt.format(
        context=citation_context,
        citation=citation_text,
        options=options_block, 
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(input_path: str, valid_labels: set[str]) -> tuple[list[dict], list[dict]]:
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
                logger.warning(f"Row {i} skipped at load time: {problems}")
                continue

            examples.append({
                "idx":              len(examples),   # compact sequential index into examples list
                "csv_row":          i,               # original row number for traceability
                "citation_text":    citation_text,
                "citation_context": citation_context,
                "label":            label,
            })

    logger.info(
        f"Loaded {len(examples)} valid examples from {input_path} "
        f"({len(invalid)} rows skipped at load time)."
    )
    if invalid:
        logger.warning(
            f"Skipped {len(invalid)} rows during loading — "
            f"see 'invalid_rows' in the output JSON for details."
        )
    return examples, invalid


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str, valid_labels: set[str]) -> str | None:
    """
    Extracts a valid label from {"category": "<label>"} JSON response.
    Falls back through regex extraction, then substring search.
    Labels sorted longest-first so 'substantiation + basis' beats 'substantiation'.
    """
    # Strategy 1: clean JSON parse
    try:
        parsed = json.loads(raw.strip())
        label  = parsed.get("category", "").strip()
        if label in valid_labels:
            return label
        for vl in valid_labels:
            if vl.lower() == label.lower():
                return vl
    except (json.JSONDecodeError, AttributeError):
        pass

    # Strategy 2: regex — handles preamble/postamble around JSON
    match = re.search(r'\{[^}]*"category"\s*:\s*"([^"]+)"[^}]*\}', raw)
    if match:
        label = match.group(1).strip()
        if label in valid_labels:
            return label
        for vl in valid_labels:
            if vl.lower() == label.lower():
                return vl

    # Strategy 3: substring fallback — model ignored JSON format entirely
    raw_lower = raw.lower()
    for vl in sorted(valid_labels, key=len, reverse=True):
        if vl.lower() in raw_lower:
            return vl

    return None


# ── Async inference ───────────────────────────────────────────────────────────

async def infer_one_async(
    client: AsyncClient,
    semaphore: asyncio.Semaphore,
    example: dict,
    prompt: str,
    valid_labels: set[str],
) -> dict:
    """Runs inference for a single example with retry logic, respecting the semaphore."""
    raw_response = ""
    parsed_label = None
    attempts     = 0   # tracked separately — 'attempt' in loop head would be unbound if MAX_RETRIES=0

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            attempts = attempt
            async with semaphore:
                try:
                    response = await client.chat(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
                    )
                    raw_response = response["message"]["content"].strip()
                    parsed_label = parse_response(raw_response, valid_labels)
                    if parsed_label is not None:
                        break
                    logger.debug(f"idx={example['idx']} attempt={attempt}: unparseable → '{raw_response}'")
                except Exception as e:
                    logger.warning(f"idx={example['idx']} attempt={attempt}: error → {e}")
                    await asyncio.sleep(2 * attempt)
    except BaseException as e:
        # Catches CancelledError and any other unexpected coroutine-level failure.
        # The result dict is still returned so its slot in results[] is filled.
        logger.error(f"idx={example['idx']}: coroutine aborted ({type(e).__name__}: {e})")

    return {
        "idx":           example["idx"],
        "csv_row":       example["csv_row"],
        "citation_text": example["citation_text"],
        "label":         example["label"],
        "prediction":    parsed_label,
        "raw_response":  raw_response,
        "attempts":      attempts,
    }


async def run_inference_async(
    examples: list[dict],
    prompts: list[str],
    valid_labels: set[str],
) -> list[dict]:
    """Fires all requests concurrently, bounded by CONCURRENCY semaphore."""
    client    = AsyncClient()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        infer_one_async(client, semaphore, ex, prompt, valid_labels)
        for ex, prompt in zip(examples, prompts)
    ]

    logger.info(f"Running {len(tasks)} inference tasks (concurrency={CONCURRENCY})...")
    start     = time.time()
    completed = 0
    results   = [None] * len(examples)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results[result["idx"]] = result
        completed += 1
        if completed % 50 == 0 or completed == len(examples):
            elapsed = time.time() - start
            rate    = completed / elapsed
            eta     = (len(examples) - completed) / rate if rate > 0 else 0
            logger.info(
                f"  {completed}/{len(examples)} done "
                f"({rate:.1f} ex/s, ETA {eta/60:.1f} min)"
            )

    # Guard: if any slot is still None (coroutine escaped without returning),
    # replace with a sentinel so downstream code never crashes on None["prediction"].
    for i, r in enumerate(results):
        if r is None:
            logger.error(f"Result slot {i} is None — coroutine did not return. Filling sentinel.")
            results[i] = {
                "idx":           i,
                "csv_row":       examples[i]["csv_row"],
                "citation_text": examples[i]["citation_text"],
                "label":         examples[i]["label"],
                "prediction":    None,
                "raw_response":  "",
                "attempts":      0,
            }

    return results


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_response_sample(results: list[dict], n: int = RESPONSE_SAMPLE):
    """Prints a random sample of raw model responses for sanity-checking."""
    sample = random.sample(results, min(n, len(results)))
    print(f"\n{'='*60}")
    print(f"  RAW RESPONSE SAMPLE ({len(sample)} of {len(results)})")
    print(f"{'='*60}")
    for r in sample:
        print(f"\n  [csv_row={r['csv_row']}  true={r['label']}  pred={r['prediction']}]")
        print(f"  citation_text : {r['citation_text'][:120]}")
        print(f"  raw_response  : {r['raw_response']}")
        print(f"  {'-'*56}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_and_save_metrics(
    results: list[dict],
    invalid_rows: list[dict],
    valid_labels: set[str],
    output_dir: Path,
    model_name: str,
):
    # Separate inference results into those with a parsed prediction and those without
    parseable = [r for r in results if r["prediction"] is not None]
    unparseable = [r for r in results if r["prediction"] is None]

    logger.info(
        f"Evaluating {len(parseable)} examples "
        f"({len(unparseable)} unparseable, {len(invalid_rows)} invalid at load time)."
    )

    if not parseable:
        logger.error("No valid predictions to evaluate.")
        return

    y_true = [r["label"]      for r in parseable]
    y_pred = [r["prediction"] for r in parseable]

    # Pin the label order to valid_labels so the confusion matrix is always
    # the full NxN grid, not a subset determined by what happened to appear
    # in this particular test set. Sort for deterministic ordering.
    labels = sorted(valid_labels)

    # Warn if any ground-truth label fell outside valid_labels — should not
    # happen after load_data validation, but makes the guarantee explicit.
    rogue_true = set(y_true) - valid_labels
    rogue_pred = set(y_pred) - valid_labels
    if rogue_true:
        logger.error(f"y_true contains labels outside valid_labels: {rogue_true}")
    if rogue_pred:
        logger.error(f"y_pred contains labels outside valid_labels: {rogue_pred}")

    macro_f1        = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    weighted_f1     = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    cm              = confusion_matrix(y_true, y_pred, labels=labels)
    report          = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    print_response_sample(results)

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"  Total loaded:    {len(results) + len(invalid_rows)}")
    print(f"  Invalid at load: {len(invalid_rows)}")
    print(f"  Ran inference:   {len(results)}")
    print(f"  Parseable:       {len(parseable)}")
    print(f"  Unparseable:     {len(unparseable)}")
    print(f"{'='*60}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall:    {macro_recall:.4f}")
    print(f"  Weighted F1:     {weighted_f1:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {labels}")
    for i, row in enumerate(cm):
        print(f"  {labels[i]:30s} {list(row)}")
    print(f"\n  Classification Report:\n{report}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace(":", "_").replace("/", "_")

    metrics_payload = {
        "model":                 model_name,
        "n_total":               len(results) + len(invalid_rows),
        "n_invalid_at_load":     len(invalid_rows),
        "n_ran_inference":       len(results),
        "n_parseable":           len(parseable),
        "n_unparseable":         len(unparseable),
        "macro_f1":              macro_f1,
        "macro_precision":       macro_precision,
        "macro_recall":          macro_recall,
        "weighted_f1":           weighted_f1,
        "labels":                labels,
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }

    metrics_path = output_dir / f"metrics_{safe_model}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Full inference results
    results_path = output_dir / f"results_{safe_model}.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Full results saved to {results_path}")

    # Rows that couldn't be parsed after all retries
    if unparseable:
        unparseable_path = output_dir / f"unparseable_{safe_model}.jsonl"
        with open(unparseable_path, "w") as f:
            for r in unparseable:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Unparseable results saved to {unparseable_path}")

    # Rows that were rejected before inference even ran
    if invalid_rows:
        invalid_path = output_dir / f"invalid_rows_{safe_model}.jsonl"
        with open(invalid_path, "w") as f:
            for r in invalid_rows:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Invalid rows saved to {invalid_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main_async():
    logger.info(f"Model:       {MODEL_NAME}")
    logger.info(f"Input data:  {INPUT_DATA_PATH}")
    logger.info(f"Output dir:  {OUTPUT_PATH}")
    logger.info(f"Prompt file: {PROMPT_FILE}")

    # 1. Load prompt config
    judge_prompt, descriptions, category_examples, valid_labels = load_prompt_config(PROMPT_FILE)

    # 2. Load and validate data — pass valid_labels so load_data can check labels at source
    examples, invalid_rows = load_data(INPUT_DATA_PATH, valid_labels)

    if not examples:
        logger.error("No valid examples to run inference on. Check your input file.")
        return

    # 3. Build prompts (options block is the same for every example — build once)
    logger.info("Building prompts...")
    options_block = build_options_block(descriptions, category_examples)
    prompts = [
        build_prompt(ex["citation_text"], ex["citation_context"], judge_prompt, options_block)
        for ex in examples
    ]

    print(f"\n{'='*60}")
    print("  FIRST PROMPT (sanity check)")
    print(f"{'='*60}")
    print(prompts[0])
    print(f"{'='*60}\n")

    # 4. Run inference
    t0      = time.time()
    results = await run_inference_async(examples, prompts, valid_labels)
    elapsed = time.time() - t0
    logger.info(f"Inference complete in {elapsed/60:.1f} min.")

    # 5. Compute and save metrics
    compute_and_save_metrics(results, invalid_rows, valid_labels, Path(OUTPUT_PATH), MODEL_NAME)


if __name__ == "__main__":
    asyncio.run(main_async())