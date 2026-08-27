import os
import pandas as pd
import sqlite3
from tqdm import tqdm
import logging

from src.utils import load_env, get_logger
from src.db_utils import add_clean_title_column, clean_title


"""
Add CSS/Non-CSS supervision to the article table.
"""

def main():
    env_vars = load_env()
    logger = get_logger("create_pred_table", level=logging.INFO)

    print("\n----------------------------------------------------")
    logger.info("STEP 1:Adding supervision labels to article table...")

    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    add_clean_title_column(db_path=env_vars["DB_PATH"], table_name="article", logger=logger)

    # read csv with labelled examples
    # conference,year,track,nlp_css,title
    df = pd.read_csv(env_vars["LABELED_PAPERS_PATH"])


    # clear all values in nlp_css column
    clear_query = "UPDATE article SET nlp_css = NULL"
    cur.execute(clear_query)

    # iterate through venues and years to get stats of which we were able to add
    grouped_df = df.groupby(["year", "conference"])
    for (year, conference), group in grouped_df:
        logger.debug(f"{conference} {year}: {len(group)} papers")
        num_labelled = len(group)
        num_failed = 0
        for i in range(len(group)):

            row = group.iloc[i]
            conference = row["conference"].lower()
            year = row["year"]
            track = row["track"]
            nlp_css = row["nlp_css"]
            title = row["title"]

            # match article id with title
            try:
                # find title match
                clean = clean_title(title)
                match_query = f"SELECT id FROM article WHERE clean_title = '{clean}' AND year = {year} AND venue = '{conference}'"
                cur.execute(match_query)
                id = cur.fetchone()[0]

                # set nlp_css value to 1 or 0
                cur.execute(f"UPDATE article SET nlp_css = {nlp_css} WHERE id = {id}")
            except:
                # add it to the table, leave abstract blank
                num_failed += 1
                continue

        logger.debug(f"Failed to match {num_failed}/{num_labelled} papers for {conference} {year}")

        # update nlp_css column
        update_query = f"UPDATE article SET nlp_css = {nlp_css} WHERE id = {id}"
        cur.execute(update_query)

        # update track column
        track = track.replace("'", "''")
        update_query = f"UPDATE article SET track = '{track}' WHERE id = {id}"
        cur.execute(update_query)


    # where the venue is nlp_css, set nlp_css = 1
    cur.execute("UPDATE article SET nlp_css = 1 WHERE venue IS 'nlp_css'")
    cur.execute("UPDATE article SET nlp_css = 1 WHERE venue IS 'sighum'")

    # print results
    pos_count = cur.execute("SELECT COUNT(*) FROM article WHERE nlp_css = 1").fetchone()[0]
    neg_count = cur.execute("SELECT COUNT(*) FROM article WHERE nlp_css = 0").fetchone()[0]

    # set everything that's not 1 or 0 to NULL
    cur.execute("UPDATE article SET nlp_css = NULL WHERE nlp_css NOT IN (0, 1)")

    logger.info(f"Successfully added supervision labels to article table, now has {pos_count} positive and {neg_count} negative examples.")

    con.commit()
    con.close()


if __name__ == "__main__":
    main()