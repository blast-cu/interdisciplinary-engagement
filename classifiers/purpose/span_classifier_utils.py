import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union
import wandb
import numpy as np
import pandas as pd
import os

from src.utils import save_json

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks, forward function edited to do span classification."""

    def __init__(self, config, span_pooling):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.span_pooling = span_pooling

    def forward(self, last_hidden, span_indices, **kwargs):

        batch_size, seq_len, hidden_size = last_hidden.shape
        
        # check that span indices exist and put them on correct device
        if span_indices is None:
            raise ValueError("span_indices must be provided (tensor shape (batch,2))")
        span_indices = span_indices.to(last_hidden.device)

        # create a mask that marks tokens within the span for each example
        token_positions = torch.arange(seq_len, device=last_hidden.device).unsqueeze(0)  # (1, seq_len)
        starts = span_indices[:, 0].unsqueeze(1)
        ends = span_indices[:, 1].unsqueeze(1)

        # create mask for span tokens
        span_mask = (token_positions >= starts) & (token_positions <= ends) # True/False matrix [batch, max_input_len + 2]
        span_mask = span_mask | (token_positions == 0)  # include CLS token in mask
        span_mask = span_mask.float().unsqueeze(-1) # convert to float and add third dimension [batch_size, max_input_size + 2, 1]
        span_len = span_mask.sum(dim=1).clamp(min=1.0) + 1 # get length of span tokens [batch_size, 1], add 1 for cls token

        # POOL
        # use mask to get desired embedding columns, sum if more than 1
        if self.span_pooling == "mean":
            # get mean of columns if more than one token
            span_sum = (last_hidden * span_mask).sum(dim=1)  # [batch_size, hidden_size]
            span_repr = span_sum / span_len  # [batch_size, hidden_size]
        else: 
            raise NotImplementedError(f"Invalid option {self.span_pooling} for pooling; only mean pooling implemented")

        x = span_repr
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaSpanForSequenceClassification(RobertaPreTrainedModel):
    """
    Roberta model that classifies based on the representation of a token span.
    Expected inputs to forward:
      - input_ids, attention_mask (standard)
      - span_indices: LongTensor of shape (batch, 2) with (start_token_idx, end_token_idx) inclusive
      - labels (optional) for computing loss
    Pooling strategy: mean over span tokens by default. You can change to 'first'/'last'/'max'.
    """
    def __init__(self, config, span_pooling="mean"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config, span_pooling)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        ids: int = None,
        span_indices = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        span_indices: LongTensor shape (batch,2) -> start, end inclusive token indices relative to tokenized input.
        """
        # outputs = self.roberta(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     return_dict=True,
        #     output_hidden_states=True, output_attentions=True
        # )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden = outputs[0]
        logits = self.classifier(last_hidden, span_indices)

        ## identical to RobertaForSequenceClassification
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DataCollatorForSpanClassification:
    """
    Pads a list of examples (dicts) using tokenizer.pad, and collates span indices as tensor.
    Expects each example to contain:
      - input_ids, attention_mask
      - labels (int)
      - span_token_indices (list/tuple of two ints)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # tokenizer.pad will pad input_ids, attention_mask, etc.
        labels = [f["labels"] for f in features] if "labels" in features[0] else None
        ids = [f["ids"] for f in features] if "ids" in features[0] else None # these have to be an int apparently

        span_indices = [f["span_token_indices"] for f in features]
        # Remove keys we will not pass to tokenizer.pad
        to_pad = [{k: v for k, v in f.items() if k not in ("labels", "span_token_indices", "ids")} for f in features]
        batch = self.tokenizer.pad(to_pad, return_tensors="pt")

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if ids is not None:
            batch["ids"] = torch.tensor(ids, dtype=torch.long)

        batch["span_indices"] = torch.tensor(span_indices, dtype=torch.long)  # (batch, 2)
        return batch
    
def test(trainer, test_tokenized, ds, model_path):
    test_results = trainer.predict(test_tokenized)
    print("Test metrics:", test_results.metrics)
    wandb.log(test_results.metrics)


    # # logits, labels = test_results.predictions, test_results.label_ids
    preds = np.argmax(test_results.predictions, axis=-1)
    preds = preds.tolist()
    labels = test_results.label_ids.tolist()
    print("Test predictions:", preds)
    print("Test labels:     ", labels)

    out_test_results = {}
    out_test_results["results"] = test_results.metrics
    out_test_results["train"] = pd.DataFrame(ds["train"].to_dict()).to_dict(orient='records')
    out_test_results["validation"] = pd.DataFrame(ds["validation"].to_dict()).to_dict(orient='records')

    out_test_results["test"] = pd.DataFrame(ds["test"].to_dict()).to_dict(orient='records')
    for idx, item in enumerate(ds["test"]):
        out_test_results["test"][idx]["prediction"] = preds[idx]

    os.makedirs(model_path, exist_ok=True)
    save_json(out_test_results, model_path, "test_results.json")


def prepare_dataset(dataset, tokenizer, text_col="text", span_col="span_text", label_col="label", max_length=256):
    """
    dataset: a huggingface dataset with columns:
      - text: the raw text
      - span_start: start char index of span (inclusive)
      - span_end: end char index of span (exclusive)
      - label: int label

    Returns: tokenized dataset with 'input_ids', 'attention_mask', 'labels', 'span_token_indices' (list [start,end])
    """
    def get_span_char_indices(text, token):
        """
        Utility to get char start/end indices of a token in text.
        Returns (start_char, end_char) where end_char is exclusive.
        """
        try:
            char_start = text.index(token)
            # print(f"found {token} in {text}\n")
        except ValueError as e:
            print(f"{token} not found in {text}") 
            raise e

        char_end = char_start + len(token) - 1
        return char_start, char_end

    def tokenize_and_map(example):
        # Tokenize with offsets to map char span -> token span
        # print(f"Text being tokenized: '{example[text_col]}'")
        # print(f"Num tokens: {len(tokenizer.tokenize(example[text_col]))}")
        tok = tokenizer(
            example[text_col],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
            return_offsets_mapping=True,
        )
        offsets = tok.pop("offset_mapping")  # list of (start_char, end_char) per token
        offsets = offsets[1:-1] # remove [0,0] appended at beginning and end
        # find first token whose offset overlaps the char start, and last token whose offset overlaps char_end-1
        start_char_idx_sub, end_char_idx_sub = get_span_char_indices(example[text_col], example[span_col])
        # print(f"Context length: {len(example[text_col])}")
        # print(f"Citation: '{example[span_col]}'")
        # print(f"Context snippet around span: '{example[text_col][max(0,start_char_idx_sub-20):end_char_idx_sub+20]}'")
        tokens = tokenizer.convert_ids_to_tokens(tok["input_ids"])
        # print(f"First 20 tokens: {tokens[:20]}")
        # print(f"Tokens around span: {tokens[max(0, start_char_idx_sub-5):end_char_idx_sub+5]}")
        token_indices_for_substring = []
        for i, (token_start_char, token_end_char) in enumerate(offsets):
            # Check for overlap or containment
            if (token_start_char >= start_char_idx_sub and token_start_char < end_char_idx_sub) or \
            (token_end_char > start_char_idx_sub and token_end_char <= end_char_idx_sub) or \
            (token_start_char < start_char_idx_sub and token_end_char > end_char_idx_sub):
                token_indices_for_substring.append(i + 1) # add one bc we removed an offset

        try:
            token_start, token_end = token_indices_for_substring[0], token_indices_for_substring[-1]
        except Exception as e:
            print(example[text_col])
            print(example[span_col])
            print(f"Token indices: {token_indices_for_substring}\n")
            print(f"Start/end chars: {start_char_idx_sub}, {end_char_idx_sub}\n")
            print(f"offsets: {offsets}\n")
            print(example)
            raise e

        tok["labels"] = example[label_col]
        tok["span_token_indices"] = [token_start, token_end]
        tok["ids"] = example["id"]

        return tok

    
    tokenized = dataset.map(
        tokenize_and_map, 
        remove_columns=dataset.column_names,
        num_proc=4,       # match your --cpus-per-task
        load_from_cache_file=True,)

    return tokenized