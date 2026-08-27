# plain jain imports
import os
import random
import numpy as np
import logging
import argparse

# the stuff for fine tuning
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from sklearn.metrics import \
    f1_score, accuracy_score, precision_score, recall_score
from transformers import RobertaTokenizerFast, \
    TrainingArguments, Trainer, RobertaForSequenceClassification
from datasets import Dataset, DatasetDict
from typing import Any, Optional, Union

# our own stuff
from src.utils import set_seed
from classifiers.track.train_utils import \
    get_labelled_data, prepare_datasets, test, DataCollatorForClassification, compute_metrics

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(args):
    """
    Main training and testing function for track classifier.
    """

    # set hyperparameters
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 2e-5

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    MODEL_NAME = "roberta-base"

    # set up environment and seed
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = "data/models"
    output_model_name = "track_classifier"
    os.makedirs(model_path, exist_ok=True)
    logger.info(f"Model path: {model_path}")

    # when using fast tokenizer, must disable parallelism
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    ### TRAINING DATA PREPARATION
    labelled_data = get_labelled_data(logger)  # get all labelled data from db
    if args.test:
        logger.info("Running in test mode with smaller dataset.")
        # select 10 positives and 10 negatives
        pos_samples = [item for item in labelled_data if item["label"] == 1][:10]
        neg_samples = [item for item in labelled_data if item["label"] == 0][:10]
        labelled_data = pos_samples + neg_samples

        TRAIN_EPOCHS = 2  # reduce epochs for testing

    labels = np.array([item["label"] for item in labelled_data])
    class_weights_array = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)

    # split data into train, val, test and prep for training
    random.shuffle(labelled_data)
    n = len(labelled_data)

    ds = DatasetDict({
        "train": Dataset.from_list(labelled_data[:int(TRAIN_SPLIT * n)]),
        "validation": Dataset.from_list(labelled_data[int(TRAIN_SPLIT * n):int((TRAIN_SPLIT + VAL_SPLIT) * n)]),
        "test": Dataset.from_list(labelled_data[int((TRAIN_SPLIT + VAL_SPLIT) * n):])
    })
    # ds_tokenized = prepare_datasets(ds, tokenizer, text_col="text", label_col="label")

    def preprocess_function(examples):
        # RoBERTa does not use token_type_ids, so the tokenizer automatically handles it
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    # Apply preprocessing to your dataset dictionary
    ds_tokenized = ds.map(preprocess_function, batched=True)

    # MODEL CREATION
    num_labels = len(np.unique(labels))  # binary classification
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

    # COLLATOR, METRICS, TRAINER SETUP
    collator = DataCollatorForClassification(tokenizer=tokenizer)
    # collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # custom trainer to incorporate class weights
    class WeightedLossTrainer(Trainer):
        loss_fct = nn.CrossEntropyLoss(weight=class_weights).to(device) # Use the pre-computed weights

        def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # Calculate custom loss
            loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=f"{model_path}/{output_model_name}-progress",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        seed=seed,
        report_to="wandb",  # enables Weights & Biases integration
        use_cpu = True if device.type == "cpu" else False
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # TRAINING AND TESTING
    logger.info("Starting training...")
    trainer.train()
    out_path = os.path.join(model_path, output_model_name)
    test(trainer, ds_tokenized["test"], ds, out_path, logger)

    # save final model
    logger.info("Saving model...")
    trainer.save_model(out_path)


if __name__ == "__main__":

    # take testing arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    args = parser.parse_args()

    main(args)