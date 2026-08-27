# -*- coding: utf-8 -*-

import dataclasses
from pybrat.parser import BratParser
import spacy
from spacy.matcher import Matcher
import os
from sklearn.model_selection import KFold, train_test_split


import pandas as pd
import re
from bs4 import BeautifulSoup

from data_utils.utils import ARG_SPLIT_DIR
from bs4 import BeautifulSoup
import pickle

nlp = spacy.load('en_core_web_sm')



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


def filter_data(data_dict: dict):

    pattern = [[{'POS': 'VERB', 'OP': '?'}, {'POS': 'ADV', 'OP': '*'}, {'OP': '*'},  {'POS': 'VERB', 'OP': '+'}]]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)

    # Add pattern to matcher
    matcher.add("verb-phrases", pattern, on_match=None)

    for index, example in data_dict.items():
        clean_entities = []
        for entity in example["annotations"]["entities"]:
            mention = entity["mention"]
            doc = nlp(mention)
            # call the matcher to find matches 
            matches = matcher(doc)
            if matches != []:
                # print("Accepted entity: ", mention)
                clean_entities.append(entity)

        example["annotations"]["entities"] = clean_entities

    return data_dict


def sentence_tokenize(data_dict: dict):

    for index, example in data_dict.items():
        text = example["text"]
        text = BeautifulSoup(text, "lxml").get_text()
        text = " ".join(text.split())
        doc = nlp(text)
        data_dict[index]["sents"] = doc

    return data_dict


def get_context(data_dict: dict):

    for index, example in data_dict.items():
        text = example["text"]

        for entity_index, entity in enumerate(example["annotations"]["entities"]):
            start = entity["spans"][0]['start']
            end = entity["spans"][0]['end']
            before_text = text[:start]
            before_text = before_text.replace("\n", " ")
            before_text = before_text.split(". ")
            before_text = before_text[-2:]
            before_text = ". ".join(before_text)

            after_text = text[end:]
            after_text = after_text.replace("\n", " ")
            after_text = after_text.split(". ")
            after_text = after_text[:2]
            after_text = ". ".join(after_text)

            mention = entity["mention"]
            mention = " ".join(mention.split())
            # print("Entity: ", mention)
            context = before_text + mention + after_text
            context = " ".join(context.split())
            context = context + "."
            context = re.sub(r'<H[0-9]>', '', context)
            context = re.sub(r'</H[0-9]>', '', context)

            # print(context)
            # print()

            if mention not in context:
                print("Entity not in context")
                print("Entity: ", mention)
                print("Context: ", context)
                print()
            
            data_dict[index]["annotations"]["entities"][entity_index]["context"] = context
            data_dict[index]["annotations"]["entities"][entity_index]["mention"] = mention


    return data_dict


def main():

    os.makedirs(ARG_SPLIT_DIR, exist_ok=True)
    # Initialize a parser.
    brat = BratParser(error="ignore")
    path = "data/argument_data/compiled_corpus"
    examples = brat.parse(path)

    examples = [*map(dataclasses.asdict, examples)]
    assert all(isinstance(x, dict) for x in examples)
    assert all(isinstance(e, dict) for x in examples for e in x["entities"])
    assert all(isinstance(e, dict) for x in examples for e in x["relations"])
    assert all(isinstance(e, dict) for x in examples for e in x["events"])
    
    example_dict = {}
    for i, e in enumerate(examples):
        index = i + 1
        example_dict[index] = {}

        if index < 10:
            str_index = f"A0{index}"
        else:
            str_index = f"A{str(index)}"

        text_file = open(f"{path}/{str_index}.txt", "r")
        example_dict[index]['text'] = text_file.read()
        example_dict[index]['annotations'] = e
        

    entity_count = [len(e["annotations"]["entities"]) for e in example_dict.values()]
    print(f"Total number of entities before cleaning: {sum(entity_count)}")

    ent_count = {}
    for ann in example_dict.values():
        for e in ann["annotations"]["entities"]:
            if e["type"] in ent_count:
                ent_count[e["type"]] += 1
            else:
                ent_count[e["type"]] = 1
    
    for k, v in ent_count.items():
        print(f"{k}: {v}")


    exit()

    # TAKEN OUT FOR TESTING ONLY
    print("Filtering data...")
    example_dict = filter_data(example_dict)
    entity_count = [len(e["annotations"]["entities"]) for e in example_dict.values()]
    print(f"Total number of entities after cleaning: {sum(entity_count)}")

    # print("Tokenizing sentences...")
    # sentence_tokenize(example_dict)

    example_dict = get_context(example_dict)
    ids = list(example_dict.keys())
    split_dict = get_split_dict(ids)

    split_table = {}
    split_table["Fold"] = []
    split_table["Train"] = []
    split_table["Validation"] = []
    split_table["Test"] = []
    for i in range(5):
        split_table["Fold"].append(i)

    pickle.dump(split_dict, open("data_utils/argument_model/split_dict.pkl", "wb"))
    pickle.dump(example_dict, open("data_utils/argument_model/verb_filter_example_dict.pkl", "wb"))
    # for i in range(5):
    #     os.makedirs(f"{ARG_SPLIT_DIR}{i}", exist_ok=True)
        
    #     print(f"Creating datasets for split {i}...")
    #     train_dataset = create_dataset(split_dict[i]['train'], example_dict)
    #     print("Saving training data...")
    #     train_dataset.save(f"{ARG_SPLIT_DIR}{i}/train")
    #     split_table["Train"].append(len(train_dataset))
    #     print(f"Saved training data for split {i}")

    #     val_dataset = create_dataset(split_dict[i]['val'], example_dict)
    #     val_dataset.save(f"{ARG_SPLIT_DIR}{i}/val")
    #     print(f"Saved validation data for split {i}")
    #     split_table["Validation"].append(len(val_dataset))

    #     test_dataset = create_dataset(split_dict[i]['test'], example_dict)
    #     test_dataset.save(f"{ARG_SPLIT_DIR}{i}/test")
    #     print(f"Saved testing data for split {i}")
    #     split_table["Test"].append(len(test_dataset))
    
    # pd.DataFrame(split_table).to_csv("data_utils/visualization/output/split_counts.csv", index=True, index_label="Fold")

    



if __name__ == "__main__":
    main()
