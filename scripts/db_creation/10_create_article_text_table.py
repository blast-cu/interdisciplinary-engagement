import sqlite3
import pandas as pd
import os
import pickle
from tqdm import tqdm

from src.utils import load_env
from src.db_creation import clean_sections, get_local_article_path


import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

"""
Script to create article_text table in the database
This table contains the introduction and conclusion text of each article
"""

def create_article_text_table(cur, con):
    cur.execute("DROP TABLE IF EXISTS article_text;")

    # create article_text table
    create_table_query = """
    CREATE TABLE article_text (
        id INTEGER PRIMARY KEY,
        introduction TEXT,
        conclusion TEXT
    );
    """
    cur.execute(create_table_query)
    con.commit()



def main():

    env_vars = load_env()

    print("\n----------------------------------------------------")
    LOGGER.info("STEP 10: Creating article_text table and adding introduction/conclusion text...")

    # read article table from the database
    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    # get all positive articles
    select_articles_query = "SELECT id, venue, year FROM article WHERE (nlp_css = 1 OR nlp_css_prediction = 1);"
    cur.execute(select_articles_query)
    pos_articles = cur.fetchall()
    print(f"Found {len(pos_articles)} articles from 2024 and 2025.")


    # create article_text table
    create_article_text_table(cur, con)

    for paper_id, venue, year in tqdm(pos_articles, desc="Adding intro/conclusion text to article_text table"):

        # get pickle path 
        file_path = get_local_article_path(env_vars["VENUE_DATA_PATH"], venue, year, paper_id, LOGGER)
        if file_path is None or not os.path.exists(file_path):
            LOGGER.warning(f"{file_path} doesn't exist")
            continue

        # load pickle
        with open(file_path, 'rb') as f:
            p = pickle.load(f)

        # extract section info, clean the section title
        sections = clean_sections(p['sections'], LOGGER)

        # add intro and conclusion to the database
        sec_labels = ['introduction', 'conclusion']
        for sec_label in sec_labels:
            # prepare section text
            sec_text = sections[sec_label]['text']
            if sec_label == sec_labels[0]:  # if first section, insert
                insert_query = f"INSERT INTO article_text (id, {sec_label}) VALUES (?, ?)"
            else:
                insert_query = f"UPDATE article_text SET {sec_label} = ? WHERE id = ?"

            try:
                cur.execute(insert_query, (paper_id, sec_text))

            except sqlite3.IntegrityError:
                # this can happen when a paper is both in pos and predicted_pos
                continue


    # print the first 5 rows of the table
    cur.execute("SELECT * FROM article_text")
    print(f"article_text table now exists with {len(cur.fetchall())} entries")

    # commit and close the connection
    con.commit()
    con.close()

if __name__ == "__main__":
    main()