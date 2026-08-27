from tqdm import tqdm
import torch
import transformers
import sys
import os
import pickle
import numpy as np

from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, classification_report

PROJECT_PATH = "data/argument_model"

from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder




def train(dataset, dev_dataset, model, loss_fn, le):
  train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

  PAT = 3
  best_f1 = 0; patience = PAT

  while patience > 0:
    train_loss = 0; train_batches = 0
    y_pred_all = []; y_gold_all = []
    pbar = tqdm(range(len(train_dataloader)))

    model.train()
    for input_ids, token_ids, attention_mask, labels in train_dataloader:
      model.zero_grad()
      logits, probas = model(input_ids, token_ids, attention_mask)
      loss = loss_fn(logits, labels)
      train_loss += loss.item()
      train_batches += 1

      _, y_pred = torch.max(probas, 1)
      y_pred_all += list(y_pred.cpu())
      y_gold_all += list(labels.cpu())

      loss.backward()
      optimizer.step()

      pbar.update(1)

    pbar.close()
    train_loss = train_loss / train_batches
    mf1 = f1_score(y_gold_all, y_pred_all, average='macro')

    loss_dev,  mf1_dev = predict(dev_dataset, model, loss_fn, le)
    if mf1_dev > best_f1:
      best_f1 = mf1_dev
      patience = PAT
      #torch.save(model.state_dict(), out_model)
    else:
      patience -= 1

    print("Train loss: {0}, macro F1: {1}".format(train_loss, mf1))


def predict(dataset, model, loss_fn, le, print_report=False):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.eval()
    pbar = tqdm(range(len(test_dataloader)))
    test_loss = 0; test_batches = 0
    y_pred_all = []; y_gold_all = []

    for input_ids, token_ids, attention_mask, labels in test_dataloader:
        logits, probas = model(input_ids, token_ids, attention_mask)
        loss = loss_fn(logits, labels)
        test_loss += loss.item()
        test_batches += 1

    _, y_pred = torch.max(probas, 1)
    y_pred_all += list(y_pred.cpu())
    y_gold_all += list(labels.cpu())

    pbar.update(1)

    pbar.close()
    test_loss = test_loss / test_batches
    mf1 = f1_score(y_gold_all, y_pred_all, average='macro')

    print("Test loss: {0}, macro F1: {1}".format(test_loss, mf1))
    if print_report:
        print(classification_report(y_gold_all, y_pred_all, target_names=le.classes_, digits=4))

    return test_loss, mf1


def get_examples(example_dict, indices):
    data = []; y = []
    for index in indices:
    #print(type(example_dict[index]))
    #print(example_dict[index].keys())
    #print(type(example_dict[index]['annotations']))
    #print(example_dict[index]['annotations'].keys())
        for example in example_dict[index]['annotations']['entities']:
            #print("Mention |", example['mention'])
            #print("Context |", example['context'])
            #print("Type |", example['type'])
            #break
            data.append(example['mention'])
            y.append(example['type'])
    #break
    return data, y

class BertClassifier(torch.nn.Module):

  def __init__(self, n_classes, bert_model_name):
    super(BertClassifier, self).__init__()
    bert_config = transformers.AutoConfig.from_pretrained(bert_model_name)
    self.bert_model = transformers.AutoModel.from_pretrained(bert_model_name, add_pooling_layer=True)
    self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(bert_config.hidden_size, n_classes)

  def forward(self, input_ids, token_ids, attention_mask):
    outputs = self.bert_model(input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_ids,
                                 position_ids=None, head_mask=None)
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)

    logits = self.classifier(pooled_output)
    probas = torch.nn.functional.softmax(logits, dim=1)
    return logits, probas


class TextDataset(torch.utils.data.Dataset):
  def __init__(self, input_ids, token_ids, attention_mask, labels):
    self.input_ids = input_ids
    self.token_ids = token_ids
    self.attention_mask = attention_mask
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    input_ids_i = self.input_ids[index]
    token_ids_i = self.token_ids[index]
    attention_mask_i = self.attention_mask[index]
    label = self.labels[index]

    return input_ids_i, token_ids_i, attention_mask_i, label
  
def prepare_dataset(X, bert_model):
    input_ids = []; token_ids = []; attention_mask = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model)
    for text in X:
        encoded_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=100)
        input_ids.append(encoded_inputs['input_ids'])
        token_ids.append(encoded_inputs['token_type_ids'])
        attention_mask.append(encoded_inputs['attention_mask'])

    input_ids = torch.LongTensor(input_ids).cuda()
    token_ids = torch.LongTensor(token_ids).cuda()
    attention_mask = torch.LongTensor(attention_mask).cuda()

    return input_ids, token_ids, attention_mask


def predict(dataset, model, loss_fn, le, print_report=False):
  test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

  model.eval()
  pbar = tqdm(range(len(test_dataloader)))
  test_loss = 0; test_batches = 0
  y_pred_all = []; y_gold_all = []

  for input_ids, token_ids, attention_mask, labels in test_dataloader:
    logits, probas = model(input_ids, token_ids, attention_mask)
    loss = loss_fn(logits, labels)
    test_loss += loss.item()
    test_batches += 1

    _, y_pred = torch.max(probas, 1)
    y_pred_all += list(y_pred.cpu())
    y_gold_all += list(labels.cpu())

    pbar.update(1)

  pbar.close()
  test_loss = test_loss / test_batches
  mf1 = f1_score(y_gold_all, y_pred_all, average='macro')

  print("Test loss: {0}, macro F1: {1}".format(test_loss, mf1))
  if print_report:
    print(classification_report(y_gold_all, y_pred_all, target_names=le.classes_, digits=4))

  return test_loss, mf1


def compute_weights(Y, output_dim):
    counts = np.bincount(Y)
    num_missing = output_dim - len(counts)
    counts = np.append(counts, np.asarray([0]*num_missing))

    # smoothing
    counts[counts == 0] = 1
    #compute weights
    weights = np.asarray(np.max(counts)*1.0/counts, dtype = "float32")
    return weights

def main():
    split_dict = pickle.load(open(os.path.join(PROJECT_PATH, 'split_dict.pkl'), 'rb'))
    example_dict = pickle.load(open(os.path.join(PROJECT_PATH, 'verb_filter_example_dict.pkl'), 'rb'))

    train_indices = split_dict[0]['train']
    val_indices = split_dict[0]['val']
    test_indices = split_dict[0]['test']
    print(len(train_indices), len(val_indices), len(test_indices))

    train_data, train_y = get_examples(example_dict, train_indices)
    val_data, val_y = get_examples(example_dict, val_indices)
    test_data, test_y = get_examples(example_dict, test_indices)

    print(len(train_y), len(val_y), len(test_y))
    print(Counter(train_y))
    print(Counter(val_y))
    print(Counter(test_y))

    train_input_ids, train_token_ids, train_attention_mask = prepare_dataset(train_data, 'bert-base-uncased')
    val_input_ids, val_token_ids, val_attention_mask = prepare_dataset(val_data, 'bert-base-uncased')
    test_input_ids, test_token_ids, test_attention_mask = prepare_dataset(test_data, 'bert-base-uncased')

    le_arguments = LabelEncoder()
    train_y = torch.LongTensor(le_arguments.fit_transform(train_y)).cuda()
    val_y = torch.LongTensor(le_arguments.transform(val_y)).cuda()
    test_y = torch.LongTensor(le_arguments.transform(test_y)).cuda()

    train_dataset = TextDataset(train_input_ids, train_token_ids, train_attention_mask, train_y)
    val_dataset = TextDataset(val_input_ids, val_token_ids, val_attention_mask, val_y)
    test_dataset = TextDataset(test_input_ids, test_token_ids, test_attention_mask, test_y)

    train_data, train_y = get_examples(example_dict, train_indices)
    val_data, val_y = get_examples(example_dict, val_indices)
    test_data, test_y = get_examples(example_dict, test_indices)

    print(len(train_y), len(val_y), len(test_y))
    print(Counter(train_y))
    print(Counter(val_y))
    print(Counter(test_y))

    cws = compute_weights(train_y.numpy(), 3)
    cws = torch.FloatTensor(cws).cuda()



    loss_fn = torch.nn.CrossEntropyLoss(weight=cws)
    model = BertClassifier(3, 'bert-base-uncased')
    model.cuda()

    train(train_dataset, val_dataset, model, loss_fn, le_arguments)
    predict(test_dataset, model, loss_fn, le_arguments, print_report=True)

if __name__ == '__main__':
    main()