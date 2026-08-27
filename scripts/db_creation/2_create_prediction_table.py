import os
import sqlite3
import json
import pandas as pd
from tqdm import tqdm

from data_utils.utils import DATA_DIR, DB_FILENAME, OLD_DB_FILENAME
from data_utils.parse_pdf.utils import clean_title

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def main():
    print("\n----------------------------------------------------")
    LOGGER.info("STEP 2: Creating predictions table...")

    # open old database, get all article ids that have entries in citations table
    old_con = sqlite3.connect(OLD_DB_FILENAME)
    old_cur = old_con.cursor()

    select_query = """
    SELECT DISTINCT article_id
    FROM article_citations
    """
    old_cur.execute(select_query)
    processed_ids = old_cur.fetchall()
    processed_ids = [p[0] for p in processed_ids]
    LOGGER.info(f"Fetched {len(processed_ids)} processed ids from old database")
    old_con.close()

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()


    delete_table_query = "DROP TABLE IF EXISTS predictions"
    cur.execute(delete_table_query)

    create_table_query = \
        "CREATE TABLE predictions(\
            article_id INTEGER PRIMARY KEY,\
            nlp_css_prediction INTEGER,\
            needs_processing INTEGER,\
            FOREIGN KEY(article_id) REFERENCES article(id)\
        )"
    cur.execute(create_table_query)

    # add new predictions
    with open(f"{DATA_DIR}results/track_predictions.json", 'r') as f:
        predictions = json.load(f)


    for pred_info in tqdm(predictions, desc="Adding predictions"):
        article_id = pred_info["id"]
        prediction = pred_info["predicted_label"]

        needs_processing = 1
        if article_id in processed_ids:
            needs_processing = 0

        try:
            insert_query = f"INSERT INTO predictions (article_id, nlp_css_prediction, needs_processing) \
                VALUES ({article_id}, {prediction}, {needs_processing})"
            cur.execute(insert_query)

        except Exception as e:
            print(insert_query)
            raise e


    check = "SELECT count(article.id) FROM article \
                LEFT JOIN predictions ON predictions.article_id = article.id WHERE article.nlp_css = 1 OR predictions.nlp_css_prediction = 1;"
    cur.execute(check)
    positives_count = cur.fetchone()[0]

    check = "SELECT count(article.id) FROM article \
                LEFT JOIN predictions ON predictions.article_id = article.id WHERE article.nlp_css = 0 OR predictions.nlp_css_prediction = 0;"
    cur.execute(check)
    negatives_count = cur.fetchone()[0]

    check_article_count = "SELECT count(*) FROM article"
    cur.execute(check_article_count)
    all_count = cur.fetchone()[0]
    LOGGER.info(f"{all_count} total articles in article table")

    if (positives_count + negatives_count) != all_count:
        raise ValueError(f"Missing some predictions: {positives_count + negatives_count} predictions, but {all_count} total articles")
    

    # check how many articles need to be processed
    check = "SELECT a.year, a.venue, count(p.article_id) FROM predictions as p \
        JOIN article as a ON p.article_id = a.id \
        WHERE p.nlp_css_prediction = 1 AND p.needs_processing = 1 \
        GROUP BY a.year, a.venue;"
    cur.execute(check)
    needs_processing_count = cur.fetchall()
    for r in needs_processing_count:
        LOGGER.debug(r)
    LOGGER.info(f"{needs_processing_count} articles still need to be processed")



    con.commit()
    con.close()

if __name__ == "__main__":
    main()