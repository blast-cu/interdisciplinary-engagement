import os
import sqlite3
import pickle
from tqdm import tqdm
import logging

from src.utils import load_env, get_logger
from src.db_utils import clean_title, create_connection
from src.db_creation import clean_string, handle_none
 

def create_database(db_filename):
    """
    create the database or open existing, drop article table
    """
    initial_count = 0
    if os.path.exists(db_filename):
        # delete it 
        os.remove(db_filename)

    con, cur = create_connection(db_filename)

    # drop a pre-existing article table
    delete_table_query = "DROP TABLE IF EXISTS article"
    cur.execute(delete_table_query)

    create_table_query = \
        "CREATE TABLE article(\
            id INTEGER PRIMARY KEY AUTOINCREMENT,\
            pdf_id INTEGER NOT NULL,\
            title TEXT NOT NULL,\
            clean_title TEXT UNIQUE,\
            abstract TEXT,\
            venue TEXT NOT NULL,\
            year INTEGER NOT NULL,\
            nlp_css INTEGER,\
            track TEXT\
        )"
    cur.execute(create_table_query)
    con.commit()

    return con, cur

def add_old_articles(cur, con, old_db_filename, logger: logging.Logger):
    """
    add entries from pre-existing db (data/dbs/old_data.db) into article table
    """
    if os.path.exists(old_db_filename):
        # create connection
        old_con = sqlite3.connect(old_db_filename)
        old_cur = old_con.cursor()
        
        select_query = "SELECT id, pdf_id, title, abstract, venue, year, nlp_css FROM article;"
        old_cur.execute(select_query)
        rows = old_cur.fetchall()
        empty_title_count = 0
        for row in tqdm(rows, desc="Adding old articles to new database"):
            title = clean_string(row[2])
            if len(title) == 0:  # skip if empty title
                empty_title_count += 1
                continue

            row = (row[0], row[1], title, clean_title(row[2]), clean_string(row[3]), clean_string(row[4]).lower(), row[5], handle_none(row[6]))
            insert_query = f"INSERT INTO article(id, pdf_id, title, clean_title, abstract, venue, year, nlp_css) VALUES{row}"
            try:
                cur.execute(insert_query)
            except Exception as e:
                logger.error(f"Error inserting old article: \n{insert_query}")
                raise e

        old_con.close()
        con.commit()

        logger.debug(f"Added {len(rows) - empty_title_count}/{len(rows)} reccords from old database")
    
    else:
        logger.debug(f"No pre-existing database to add reccords from")

def handle_duplicate_article():
    return None

def add_pickle(file_name: str, file_path: str, venue: str, year: int, nlp_css: int, cur: sqlite3.Cursor, logger: logging.Logger):
    # load data from pickle
    with open(f"{file_path}/{file_name}", "rb") as f:
        data = pickle.load(f)
    
    # clean data for insertion
    title = clean_string(data["title"])
    if len(title) == 0:  # skip
        return 1
    title_clean = clean_title(title)
    pdf_id = int(file_name.split(".")[0])
    
    abstract = clean_string(data["abstract"].strip())
    if abstract.endswith(". 1"):
        abstract = abstract[:-2]

    row = (pdf_id, title, title_clean, abstract, venue.lower(), year, nlp_css)
    insert_query = f"INSERT INTO article(pdf_id, title, clean_title, abstract, venue, year, nlp_css) VALUES{row}"
    try:
        cur.execute(insert_query)

    except sqlite3.IntegrityError as e:
        # handle duplicate 
        # check if same venue and year
        get_old_query = f"SELECT venue, year FROM article  WHERE clean_title = '{title_clean}'"
        cur.execute(get_old_query)
        old_venue, old_year  = cur.fetchone()
        if old_venue == venue and old_year == year:
            # update nlp + css label 
            update_query = f"UPDATE article SET nlp_css = {nlp_css} WHERE clean_title = '{title_clean}'"
            cur.execute(update_query)
        else:
            logger.error(f"Integrity error for {title_clean} in {venue} {year}: {e}")
            raise e

    return 0
    
def add_unlabelled_abstracts(pickle_file: str, venue: str, year: int, cur: sqlite3.Cursor, logger: logging.Logger):
    """
    Add unlabelled abstracts for main conference from pickle file
    """
    invalid_title_count = 0
    if not os.path.exists(pickle_file):
        logger.warning(f"{pickle_file} does not exist, skipping...")
        return 0

    unlabelled_abstracts = pickle.load(open(pickle_file, "rb"))
    for pdf_id, abstract in unlabelled_abstracts.items():

        title = abstract['title']
        if len(title) == 0:
            invalid_title_count += 1
            continue

        title = title.replace("'", "''")
        try:
            abstract = abstract['abstract']
        except:
            logger.warning(f"Abstract not found for {title}")
            continue
        abstract = abstract.replace("'", "''")
        title_clean = clean_title(title)

        insert_query = \
            f"INSERT INTO article(clean_title, pdf_id, title, abstract, venue, year)\
                VALUES('{title_clean}', {pdf_id}, '{title}', '{abstract}', '{venue.lower()}', {year})"

        try:
            cur.execute(insert_query)
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: article.clean_title" in str(e):
                # keep existing, probably has more data
                logger.debug(f"Duplicate article title found for unlabelled abstract: {title_clean}, skipping...")
                pass
            else:
                logger.error(f"Integrity error for {title_clean} in {venue} {year}: {e}")
                raise e
        
    return invalid_title_count



def main():

    env_vars = load_env()
    logger = get_logger("create_art_tab", level=logging.INFO)

    print("----------------------------------------------------")
    logger.info("STEP 0: Creating article table...")

    # create database or open existing
    con, cur = create_database(env_vars["DB_PATH"])

    # if there's an old database, add records from there
    add_old_articles(cur, con, env_vars["OLD_DB_PATH"], logger)

    # iterate through the directory and add the data to the database
    venue_dir = env_vars["VENUE_DATA_PATH"]
    invalid_title_count = 0
    for venue in tqdm(os.listdir(venue_dir), desc="Processing venue directories"):  # directories for each venue
        venue_path = venue_dir + venue

        if os.path.isdir(venue_path):  # ensure it's a directory
            for year in os.listdir(venue_path):
                year_path = venue_path + "/" + year
                if os.path.isdir(year_path):  # ensure it's a directory
                    year = int(year)

                    # ADD LABELED DATA
                    if "nlp_css" in venue_path or "sighum" in venue_path:  # dedicated workshops, all positive
                        pickle_dir = f"{year_path}/dict/"

                        if os.path.isdir(pickle_dir):
                            # process files in base directory
                            nlp_css = 1
                            for file in os.listdir(pickle_dir):
                                if file.split(".")[-1] == "pkl":
                                    invalid_title_count += add_pickle(file, pickle_dir, venue, year, nlp_css, cur, logger)
                                else:
                                    logger.warning(f"{file} is not a pickle file, skipping...")
                        else:
                            logger.warning(f"{pickle_dir} is not a directory, skipping...")
                    
                    else: # labelled data from full conferences
                        for label in ["pos", "neg"]:
                            pickle_dir = f"{year_path}/{label}/dict/"

                            if os.path.isdir(pickle_dir):
                                nlp_css = 1 if label == "pos" else 0

                                for file in os.listdir(pickle_dir):
                                    if file.split(".")[-1] == "pkl":
                                        invalid_title_count += add_pickle(file, pickle_dir, venue, year, nlp_css, cur, logger)
                                    else:
                                        logger.warning(f"{file} is not a pickle file, skipping...")
                            else:
                                logger.warning(f"{pickle_dir} is not a directory, skipping...")

                        # ADD UNLABELLED DATA
                        # not needed for dedicated workshops
                        unlabelled_abstracts = f"{year_path}/unlabelled/abstracts.pkl"
                        invalid_title_count += add_unlabelled_abstracts(unlabelled_abstracts, venue, year, cur, logger)
                        
        else:
            logger.warning(f"{venue_path} is not a directory, skipping...")

    logger.warning(f"Failed to add {invalid_title_count} entries due to empty titles")

    cur.execute("SELECT COUNT(*) FROM article")
    logger.info(f"  final article table contains {cur.fetchone()[0]} rows")

    con.commit()
    con.close()

if __name__ == "__main__":
    main()