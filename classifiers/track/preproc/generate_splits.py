import os
from sklearn.model_selection import KFold, train_test_split
import sqlite3
import random
import re
import string
from transformers import RobertaTokenizer

from data_utils.model_utils.dataset import ClassificationDataset
from data_utils.utils import SPLIT_DIR, DB_FILENAME

def clean_text(text):
    # to lower case
    text = text.lower()
    # remove links
    text = re.sub('https:\/\/\S+', '', text) 
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    # remove next line     
    text = re.sub(r'[^ \w\.]', '', text) 
    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    return text


def get_ids():
    """
    Get a list of article IDs from the database.

    Returns:
        list: A list of article IDs.
    """
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    cur.execute("SELECT id FROM article where nlp_css is not null")
    ids = [row[0] for row in cur.fetchall()]
    print(f"Number of articles: {len(ids)}")
    return ids


def get_split_dict(ids: list, n_splits: int = 5):
    """
    Generate a dictionary of train/test splits for a given quality dictionary.

    Args:
        qual_dict (dict): A dictionary containing quality information for articles.

    Returns:
        dict: A dictionary of train/test splits, where each key represents a fold number and the value is a dictionary
              containing the train and test article IDs for that fold.
    """
    split_dict = {}
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    split = kf.split(ids)

    # format split dictionary {key = fold number, value = dict of train/test ids}
    for i, (train_index, test_index) in enumerate(split):
        split_dict[i] = {}
        train_ids = [ids[t] for t in train_index]
        train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
        split_dict[i]['train'] = train_ids
        split_dict[i]['val'] = val_ids
        split_dict[i]['test'] = [ids[t] for t in test_index]

    return split_dict

def create_dataset(ids: list, balance: bool = False):
    """
    Query db to create a dataset from a list of article IDs.

    Args:
        ids (list): A list of article IDs.

    Returns:
        ClassificationDataset: A ClassificationDataset object containing the data from the given article IDs.
    """
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    abstracts = []
    labels = []
    tokenizer = RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path='roberta-large',
                         problem_type="single_label_classification")

    for id in ids:
        cur.execute(f"SELECT abstract, nlp_css, venue, year, pdf_id FROM article WHERE id = {id}")
        row = cur.fetchone()
        if row:
            text = clean_text(row[0])
            if len(text) == 0:
                # print(f"ID {id} has no abstract.")
                # print(row)
                # print()
                continue
            abstracts.append(row[0])
            labels.append(row[1])
        else: 
            print(f"ID {id} not found in database.")

    if balance:
        print("Balancing dataset")
        
        neg_examples = [i for i, label in enumerate(labels) if label == 0]
        pos_examples = [i for i, label in enumerate(labels) if label == 1]
        pos_count = labels.count(1)

        print(f"Number of positive examples: {pos_count}")

        new_neg_examples = neg_examples[:pos_count]
        print(f"Number of negative examples: {len(new_neg_examples)}")
        new_labels = [0] * pos_count + [1] * pos_count
        new_abstracts = [abstracts[i] for i in new_neg_examples] + [abstracts[i] for i in pos_examples]
        new_zip = list(zip(new_abstracts, new_labels))

        print(f"Number of examples: {len(new_abstracts)}")

        random.shuffle(new_zip)
        abstracts, labels = zip(*new_zip)

    abstracts = list(abstracts)
    labels = list(labels)


    dataset = ClassificationDataset(ids, abstracts, labels, tokenizer)
    return dataset


def main():

    ids = get_ids()  # list of article IDs
    split_dict = get_split_dict(ids)

    for i in range(5):
        os.makedirs(f"{SPLIT_DIR}{i}", exist_ok=True)

        train_dataset = create_dataset(split_dict[i]['train'])
        train_dataset.save(f"{SPLIT_DIR}{i}/train")  # we'll have to see about this

        val_dataset = create_dataset(split_dict[i]['val'])
        val_dataset.save(f"{SPLIT_DIR}{i}/val")  # we'll have to see about this

        test_dataset = create_dataset(split_dict[i]['test'])
        test_dataset.save(f"{SPLIT_DIR}{i}/test")  # we'll have to see about this
    
if __name__ == "__main__":
    main()