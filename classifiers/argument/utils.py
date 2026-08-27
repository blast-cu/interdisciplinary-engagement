import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
import os
from transformers import RobertaTokenizerFast
from data_utils.model_utils.dataset import ArgClassificationDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import cross_entropy
from sklearn.metrics import f1_score


label_map = {
    'background_claim': 0,
    'data': 1,
    'own_claim': 2
}

def get_weights(y, classes: list=[0, 1, 2]):
    """
    Computes class weights for weighted loss.

    Parameters:
    - y: list or array of target values

    Returns:
    - class_weights: tensor of class weights, to be used in loss function
    """
 
    if len(classes) != len(set(y)):
        y = y + classes
    classes = np.array(classes)
    weight = \
        compute_class_weight(class_weight='balanced', classes=classes, y=y)

    weight = [float(w) for w in weight]
    class_weights = torch.tensor(weight).to('cuda')
    return class_weights

class ArgClassificationDataset(Dataset):
    def __init__(self, texts, labels=[], ids=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.indicator_texts = []
        self.texts = []
        if texts:
            self.indicator_texts = [t[1] for t in texts]
            self.texts = [t[0] for t in texts]
        self.labels = labels

        self.article_ids = ids
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_indices = []
        self.end_indices = []
        self.input_ids = []
        self.attention_masks = []

        if texts and tokenizer:
            for idx, text in enumerate(self.texts):
                # print(f"Processing text {idx + 1} of {len(self.texts)}")

                indicator_text = self.indicator_texts[idx]

                temp_encoding = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=False,
                    return_offsets_mapping=True
                )

                start_index, end_index = \
                    self.get_indicator_indices(indicator_text,
                                            temp_encoding,
                                            text)

                if start_index is None or end_index is None:
                    print('Substring: ' + indicator_text)
                    print('Original text: ' + text)
                    print('Start index: ' + str(start_index))
                    print('End index: ' + str(end_index))
                    print('Offset mapping: \n')
                    for i, token in enumerate(temp_encoding['offset_mapping'][0]):
                        print(f"Token {i}: {token}")
                    raise Exception('Could not find indicator text in excerpt')

                if end_index > self.max_length:
                    text_start = end_index + int(self.max_length / 2) - self.max_length
                    text = text[text_start:]

                    start_index = start_index - text_start
                    end_index = end_index - text_start

                excerpt_encoding = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True
                )

                self.start_indices.append(start_index)
                self.end_indices.append(end_index)
                self.input_ids.append(excerpt_encoding['input_ids'].flatten())
                self.attention_masks.append(excerpt_encoding['attention_mask'].flatten())

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.texts)
    
    def get_indicator_indices(self, indicator_text, excerpt_encoding, text):
        """
        Get the start and end indices of the indicator text within the given tokenized text.

        Parameters:
            indicator_text (str): The indicator text to search for.
            excerpt_encoding (dict): The encoding of the excerpt text.
            text (str): The full text to search within.

        Returns:
            tuple: A tuple containing the start and end indices of the indicator text within the text.
        """
        sub_start = text.find(indicator_text)
        sub_end = sub_start + len(indicator_text)
        offset_map = excerpt_encoding['offset_mapping'].tolist()

        # print(indicator_text)
        # print(sub_start)
        # print(sub_end)

        start_index = None
        end_index = None
        for i, token in enumerate(offset_map[0]):
            token_start = token[0]
            token_end = token[1]
            if start_index is None:
                if sub_start >= token_start and sub_start <= token_end:
                    start_index = i
                    if sub_end <= token_end:
                        end_index = i + 1
                        break
            else:
                if sub_end >= token_start and sub_end <= token_end:
                    end_index = i + 1
                    break

        # print(start_index)
        # print(end_index)

        return start_index, end_index
    
    def save(self, path):
        """
        Saves the dataset to a file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self, path)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized  item from the dataset
        """
        text = self.texts[idx]
        if len(self.labels) > 0:
            label = self.labels[idx]
        else:
            label = -1

        if len(self.article_ids) > 0:
            article_id = self.article_ids[idx]
        else:
            article_id = -1


        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        start = self.start_indices[idx]
        end = self.end_indices[idx]

        return {
            'id': torch.tensor(idx),
            'start': start,
            'end': end,
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'label': torch.tensor(label),
            'article_ids': torch.tensor(article_id),

        }
    
    def get_labels(self):
        """
        Returns the labels of the dataset
        """
        return self.labels
    
    def load(self, path):
        """
        Loads a dataset from a file
        """
        self = torch.load(path)
        return self

class ArgModel(nn.Module):
    """
    QuantModel is a PyTorch module that implements a quantification model based on the RoBERTa architecture.

    Args:
        model_checkpoint (str): The path or identifier of the pre-trained RoBERTa model.
        num_labels (int): The number of labels for the classification task.

    Attributes:
        config (RobertaConfig): The configuration of the RoBERTa model.
        dense (Linear): The linear layer for the dense connections.
        dropout (Dropout): The dropout layer for regularization.
        out_proj (Linear): The linear layer for the output projection.
        linear (Linear): The linear layer for combining CLS and indicator tokens.
        activation (ReLU): The activation function.
        roberta (RobertaModel): The pre-trained RoBERTa model.
        softmax (Softmax): The softmax function for probability calculation.

    Methods:
        from_pretrained(path, task): Loads the pre-trained model weights from the specified path.
        save(path, task): Saves the model weights to the specified path.
        forward(indices, excerpts, excerpt_attention_mask): Performs forward pass through the model.

    """

    def __init__(self, model_checkpoint, num_labels):
        super(ArgModel, self).__init__()

        self.config = RobertaConfig.from_pretrained(model_checkpoint)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size).to('cuda')
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout).to('cuda')
        self.out_proj = nn.Linear(self.config.hidden_size, num_labels).to('cuda')

        hidden_size = self.config.hidden_size  # num values per token
        self.linear = nn.Linear(hidden_size * 3, hidden_size).to('cuda')  # n size = [1536, 2]
        self.activation = nn.ReLU().to('cuda')
        self.roberta = RobertaModel.from_pretrained(model_checkpoint).to('cuda')
        self.softmax = nn.Softmax(dim=0).to('cuda')

    def from_pretrained(self, path, task):
        """
        Loads the pre-trained model weights from the specified path.

        Args:
            path (str): The path where the model weights are saved.
            task (str): The task name used as a suffix for the weight files.

        Returns:
            QuantModel: The QuantModel instance with loaded weights.

        """
        rob_path = os.path.join(path, task + '_roberta') 
        self.roberta = RobertaModel.from_pretrained(rob_path)

        dense_path = os.path.join(path, task + '_dense')
        self.dense.load_state_dict(torch.load(dense_path))
                                   
        dropout_path = os.path.join(path, task + '_dropout')
        self.dropout.load_state_dict(torch.load(dropout_path))

        out_proj_path = os.path.join(path, task + '_out_proj')
        self.out_proj.load_state_dict(torch.load(out_proj_path))

        linear_path = os.path.join(path, task + '_linear')
        self.linear.load_state_dict(torch.load(linear_path))

        activation_path = os.path.join(path, task + '_activation')
        self.activation.load_state_dict(torch.load(activation_path))

        return self
    
    def save(self, path, task):
        """
        Saves the model weights to the specified path.

        Args:
            path (str): The path where the model weights will be saved.
            task (str): The task name used as a suffix for the weight files.

        """
        rob_path = os.path.join(path, task + '_roberta')
        self.roberta.save_pretrained(rob_path)

        dense_path = os.path.join(path, task + '_dense')
        torch.save(self.dense.state_dict(), dense_path)

        dropout_path = os.path.join(path, task + '_dropout')
        torch.save(self.dropout.state_dict(), dropout_path)

        out_proj_path = os.path.join(path, task + '_out_proj')
        torch.save(self.out_proj.state_dict(), out_proj_path)

        linear_path = os.path.join(path, task + '_linear')
        torch.save(self.linear.state_dict(), linear_path)

        activation_path = os.path.join(path, task + '_activation')
        torch.save(self.activation.state_dict(), activation_path)

    def forward(self,
                start: torch.Tensor,  # size = 1
                end: torch.Tensor,  # size = 1
                excerpts: torch.Tensor,  # size = [8, 514](ish)
                excerpt_attention_mask: torch.Tensor  # size = [8, 512]
                ):
        """
        Performs forward pass through the model.

        Args:
            indices (Tensor): The indices tensor of shape [batch_size, sequence_length, num_values].
            excerpts (Tensor): The excerpts tensor of shape [batch_size, sequence_length].
            excerpt_attention_mask (Tensor): The attention mask tensor of shape [batch_size, sequence_length].

        Returns:
            Tensor: The logits tensor of shape [batch_size, num_labels].

        """
        batch_size = excerpts.shape[0]
        rob_out = self.roberta(excerpts, attention_mask=excerpt_attention_mask)
        print(start)
        last_layer = rob_out.last_hidden_state  # size = [8, 512, 768]
        # start_indicator = torch.index_select(last_layer, 0, start)
        # end_indicator = last_layer[:, end, :]
        cls = last_layer[:, 0, :]  # size = [8, 768], CLS token
        print(cls.shape)
        print(start_indicator.shape)
        print(end_indicator.shape)

        # indicator_token = torch.cat((start_indicator, end_indicator), 3)  # size = [8, 1536]
        # print(indicator_token.shape)
        indicator_token = start_indicator

        lin_in = torch.cat((cls, indicator_token), 1)  # size = [8, 2304]

        lin_out = self.linear(lin_in)
        activation = self.activation(lin_out)

        x = self.dropout(activation)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


def check_done(val_f1_history: list, val_f1, history_len):
    """
    Check if the model has stopped improving based on the validation loss history.

    Parameters:
    - val_loss_history (list): A list of previous validation losses.
    - val_loss (float): The current validation loss.
    - patience (float): The minimum difference between the current validation loss and the previous one to consider improvement.
    - history_len (int): The maximum length of the validation loss history.

    Returns:
    - improving (bool): True if the model is still improving, False otherwise.
    - val_loss_history (list): The updated validation loss history.
    """
    improving = True
    if len(val_f1_history) == history_len:
        val_f1_history.pop(0)  # remove at index 0

        if val_f1_history[0] == val_f1_history[1]:
            if val_f1_history[1] == val_f1:
                improving = False

    val_f1_history.append(val_f1)
    return improving, val_f1_history

def create_dataset(ids: list, data_dict: dict):
    """
    Query db to create a dataset from a list of article IDs.

    Args:
        ids (list): A list of article IDs.

    Returns:
        ClassificationDataset: A ClassificationDataset object containing the data from the given article IDs.
    """

    labels = []
    texts = []
    tokenizer = RobertaTokenizerFast\
        .from_pretrained(pretrained_model_name_or_path='roberta-base',
                         problem_type="single_label_classification")
    entity_ids = []
    curr_entity_id = 0
    for id in ids:
        for entity in data_dict[id]['annotations']["entities"]:
            texts.append((entity['context'], entity['mention']))
            labels.append(label_map[entity['type']])
            entity_ids.append(curr_entity_id)
            curr_entity_id += 1

    dataset = ArgClassificationDataset(texts, labels, entity_ids, tokenizer)
    return dataset


def setup(train_ids: list,
          val_ids: list,
          test_ids: list,
          data_dict: dict,
          lr=2e-5,
          model_checkpoint: str = "roberta-base",
          split_num: int = 0):
    """
    Set up the training, validation, and testing data loaders, tokenizer, model, optimizer, and return them.

    Args:
        train_texts (list): List of training texts.
        test_texts (list): List of testing texts.
        train_labels (list): List of training labels.
        test_labels (list): List of testing labels.
        annotation_map (dict): Mapping of annotation labels to their corresponding indices.
        lr (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        model_checkpoint (str, optional): Pretrained model checkpoint. Defaults to "roberta-base".

    Returns:
        tuple: A tuple containing the model, train loader, validation loader, test loader, and optimizer.
    """
    torch.manual_seed(42)  # Set random seed for reproducibility

    tokenizer = RobertaTokenizerFast\
        .from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                         problem_type="single_label_classification")

    save_path = f"data/argument_model_datasets/{split_num}/"
    train_data = create_dataset(train_ids, data_dict)
    train_data.save(save_path + "train")
    # train_data = ArgClassificationDataset(None, None, None, None)
    # train_data = train_data.load(save_path + "train")

    val_data = create_dataset(val_ids, data_dict)
    val_data.save(save_path + "val")
    # val_data = ArgClassificationDataset(None, None, None, None)
    # val_data = train_data.load(save_path + "val")

    test_data = create_dataset(test_ids, data_dict)
    test_data.save(save_path + "test")
    # test_data = ArgClassificationDataset(None, None, None, None)
    # test_data = train_data.load(save_path + "test")


    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    cw_labels = train_data.labels + val_data.labels
    class_weights = get_weights(cw_labels)

    # Define model
    num_labels = len(set(label_map.values()))
    # model = ArgModel(model_checkpoint, num_labels=num_labels).to('cuda')
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to('cuda')

    # Define optimizer and loss function
    optimizer = torch.optim\
        .AdamW(model.parameters(), lr=lr)

    return model, train_loader, val_loader, test_loader, optimizer, class_weights


def validate(model,
             val_loader):
    """
    Evaluate the performance of the model on the validation set.

    Args:
    - model (torch.nn.Module): The trained model to be evaluated.
    - val_loader (torch.utils.data.DataLoader): The validation data loader.
    - class_weights (torch.Tensor): The weights to be applied to each class during loss calculation.

    Returns:
    - float: The average validation loss.
    """

    model.eval()
    with torch.no_grad():

        predicted_labels = []
        true_labels = []

        for batch in val_loader:

            # span = batch['span'].to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(input_ids,
                            attention_mask)

            _, predicted = torch.max(outputs.logits, 1)

            predicted_labels += predicted.tolist()
            true_labels += labels.tolist()

        f1 = f1_score(true_labels,
                      predicted_labels,
                      average='macro')

    return f1


def train(model,
          train_loader,
          val_loader,
          optimizer,
          class_weights):
    """
    Trains the given model using the provided training data and optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        class_weights (torch.Tensor): The weights for each class in the classification task.

    Returns:
        torch.nn.Module: The trained model.
    """

    improving = True
    val_f1_history = []

    history_len = 3
    epoch = 0
    
    while epoch < 15:
        model.train()
        for batch in train_loader:

            # start = batch['start'].to('cuda')
            # end = batch['end'].to('cuda')
            

            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(
                            input_ids,
                            attention_mask,
                            labels=labels
                            )

            class_weights = class_weights.to('cuda')
            loss = cross_entropy(outputs.logits, labels, weight=class_weights)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_f1 = validate(model, val_loader)

        improving, val_f1_history = check_done(val_f1_history,
                                                val_f1,
                                                history_len)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val F1: {val_f1_history[-1]:.4f}")
        epoch += 1

    return model


def test(model, test_loader):
    """
    Evaluate the performance of a given model on a test dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
    - Tuple[List[int], List[int], float]: A tuple containing the true labels, predicted labels, and test accuracy.
    """
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        out_predicted = []
        out_labels = []
        out_ids = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            ids = batch['id']

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            out_ids += ids.tolist()
            out_predicted += predicted.tolist()
            out_labels += labels.tolist()

        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")

        # return macro f1 score
        test_f1 = f1_score(out_labels, out_predicted, average='macro')
        print(f"Test F1: {test_f1:.4f}")

        return out_ids, out_labels, out_predicted