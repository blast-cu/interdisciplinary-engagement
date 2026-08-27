import sqlite3
import numpy as np
import argparse
import json
import os
import torch

from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizerFast, \
    TrainingArguments, Trainer, RobertaForSequenceClassification
from datasets import DatasetDict, Dataset

from classifiers.track.train_utils import format_abstract, compute_metrics
from data_utils.utils import DB_FILENAME, DATA_DIR
from src.utils import save_to_json

# set up logging
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "/projects/alle5715/cross_disciplinary_engagement/data/models/track_classifier/"

def get_unlabelled(
        db_path: str
) -> list:
    """
    Get the unlabelled abstracts from the database.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get the unlabelled abstracts
    cur.execute("SELECT id, title, abstract FROM article WHERE nlp_css is NULL AND abstract IS NOT NULL")
    unlabelled_abstracts = cur.fetchall()
    LOGGER.info(f"Fetched {len(unlabelled_abstracts)} articles to label")
    conn.close()

    unlabelled_abstracts = [{"id": row[0], "text": format_abstract(row[1], row[2])} for row in unlabelled_abstracts]
    return unlabelled_abstracts


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # LOAD DATA
    # get unlabelled data
    to_predict = get_unlabelled(DB_FILENAME)
    if args.test:
        to_predict = to_predict[:100]  # smaller set for testing

    # load test samples from json
    test_path = f"{MODEL_PATH}test_results.json"
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        test_og_results = test_data["results"]
        test_data = test_data["test"]

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # prep dataset 
    ds = DatasetDict({
        "predict": Dataset.from_list(to_predict),
        "test": Dataset.from_list(test_data)  # add a sanity check
    })
    # tokenized_datasets = prepare_datasets(ds, tokenizer, text_col="text", label_col=None, max_length=512)

    def preprocess_function(examples):
        # RoBERTa does not use token_type_ids, so the tokenizer automatically handles it
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    # Apply preprocessing to your dataset dictionary
    tokenized_datasets = ds.map(preprocess_function, batched=True)

    # Load a pretrained model (replace with your fine-tuned model path if available)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(device)

    # Define dummy training arguments. We only need them to instantiate the Trainer
    # The output_dir is where the trainer will save checkpoints/logs (can be a temporary directory)
    training_args = TrainingArguments(
        output_dir="./data/results",
        per_device_eval_batch_size=16, # Adjust batch size based on your VRAM
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        # You do not need to provide train_dataset or eval_dataset for prediction only
    )


    # first check that the results on the test set match
    LOGGER.info("Generating predictions on original test set to verify performance...")
    test_predictions = trainer.predict(tokenized_datasets["test"])
    metrics = test_predictions.metrics
    if metrics["test_weighted_f1"] != test_og_results["test_weighted_f1"]:
        LOGGER.error(f"Weighted f1 doesn't match original results: {metrics["test_weighted_f1"]} != {test_og_results["test_weighted_f1"]}")
        exit()
    else:
        LOGGER.info("Confirmed that old results match new ones")

    # Run predictions
    LOGGER.info("Generating predictions over all unlabelled samples....")
    predictions = trainer.predict(tokenized_datasets["predict"])

    # The 'predictions' field contains the raw logits
    logits = predictions.predictions 

    # Apply softmax to logits and get the index of the highest probability
    predicted_labels = np.argmax(logits, axis=1)

    print(f"Predicted labels shape: {predicted_labels.shape}")
    print(f"Sample predicted labels: {predicted_labels[:10]}")

    for idx, p_label in enumerate(predicted_labels):
        to_predict[idx]["predicted_label"] = int(p_label)

    out_path = "data/results/"
    out_filename = "track_predictions.json"
    if args.test:
        out_filename = out_filename.replace(".json", "-test.json")

    LOGGER.info(f"Saving {len(to_predict)} predictions...")
    save_to_json(to_predict, out_path, out_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict track labels for unlabelled abstracts.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with smaller dataset.")
    args = parser.parse_args()

    main(args)