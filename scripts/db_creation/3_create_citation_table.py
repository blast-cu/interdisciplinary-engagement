import os
import sqlite3
import pickle
from tqdm import tqdm

from src.utils import load_env, get_logger
from src.db_utils import clean_title, create_connection, add_clean_title_column, get_column_names
from src.db_creation import \
    clean_string, handle_none, fix_nulls, get_local_article_path, add_citations_to_db

import logging


def create_tables(db_filename, logger: logging.Logger):

    con, cur = create_connection(db_filename)
    # check if the tables already exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='citations'")
    citation_table_exists = cur.fetchone() is not None
    if citation_table_exists:
        cur.execute("SELECT COUNT(*) FROM article_citations")
        initial_ac_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM citations")
        initial_c_count = cur.fetchone()[0]
        logger.info(f"Adding to existing database with {initial_ac_count} article_citation records and {initial_c_count} citation records at {db_filename}")

    else:
        create_c_table_query = \
            "CREATE TABLE citations(\
                clean_title TEXT, \
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                title TEXT NOT NULL,\
                id_type TEXT NOT NULL,\
                external_id TEXT NOT NULL,\
                authors TEXT NOT NULL,\
                f_author_ln TEXT NOT NULL,\
                year INT,\
                journal TEXT,\
                meeting TEXT,\
                field_of_study TEXT,\
                fos_source TEXT,\
                clean_id TEXT,\
                UNIQUE(clean_title, year)\
            )"
        cur.execute(create_c_table_query)
        create_ac_table_query = \
            "CREATE TABLE article_citations(\
                article_id INT NOT NULL,\
                citation_id INT NOT NULL,\
                count INT NOT NULL, \
                alpha_idx TEXT,\
                nav_id TEXT, \
                FOREIGN KEY(article_id) REFERENCES article(id),\
                FOREIGN KEY(citation_id) REFERENCES citations(id),\
                PRIMARY KEY(article_id, citation_id)\
            )"
        cur.execute(create_ac_table_query)
        con.commit()
        logger.info(f"Created new citation tables at {db_filename}")
    
    # add clean_title column to citations table 
    add_clean_title_column(db_filename, table_name="citations", logger=logger)
    con.close()



def main():

    env_vars = load_env()
    logger = get_logger("create_citation_table", level=logging.INFO)

    print("\n----------------------------------------------------")
    logger.info("STEP 3: Creating citation tables and adding citations to database...")

    con, cur = create_connection(env_vars["DB_PATH"])

    # for now, drop existing tables and recreate
    delete_table_query = "DROP TABLE IF EXISTS article_citations"
    cur.execute(delete_table_query)
    delete_table_query = "DROP TABLE IF EXISTS citations"
    cur.execute(delete_table_query)
    create_tables(env_vars["DB_PATH"], logger)

    # get all articles which are either labelled nlp_css = 1 
    get_articles_query = \
        "SELECT article.id, article.venue, article.year, article.nlp_css FROM article \
        LEFT JOIN predictions ON article.id = predictions.article_id \
        WHERE ((article.nlp_css = 1) OR (predictions.nlp_css_prediction = 1));"
    
    cur.execute(get_articles_query)
    articles = cur.fetchall()

    for article in tqdm(articles, desc="Adding citations for articles"):
        article_id = article[0]
        venue = article[1]
        year = article[2]

        local_article_path = get_local_article_path(env_vars["VENUE_DATA_PATH"], venue, year, article_id, logger)
        if local_article_path is not None:
            add_citations_to_db(cur, local_article_path, article_id, year, logger)

    # if anly column contains 'NULL' strings, replace with actual NULL values
    fix_nulls(con, cur)
    con.commit()
    con.close()



if __name__ == "__main__":
    main()