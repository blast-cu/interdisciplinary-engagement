import pandas as pd
import logging
from tqdm import tqdm
import numpy as np

from src.utils import load_env, get_logger
from src.db_utils import create_connection, clean_title
from src.db_creation import clean_string


def main():

    # article_title = "Aye or naw, whit dae ye hink? Scottish independence and linguistic identity on social media"
    article_title = "Inferring latent attributes of Twitter users with label regularization" # doesn't have o'connor citation
    
    db_title = clean_title(article_title)

    logger = get_logger("match_ann_ids")
    env_vars = load_env()

    con, cur = create_connection(env_vars["DB_PATH"])

    # get info from article table
    select_art_info_query = f"SELECT id, venue, year, nlp_css, nlp_css_prediction FROM article WHERE clean_title = '{db_title}';"
    res = cur.execute(select_art_info_query).fetchone()
    logger.info(res)

    # get info from citations table
    select_citation_info_query = f"SELECT citation_text, paragraph_context FROM article_citations_context WHERE article_id = {res[0]};"
    res = cur.execute(select_citation_info_query).fetchall()

    for r in res:
        print(r)
        print()

    con.close()


if __name__ == "__main__":
    main()
