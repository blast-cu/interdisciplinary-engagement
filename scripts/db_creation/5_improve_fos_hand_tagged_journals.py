import os
import sqlite3
import pickle
import pandas as pd
import math

SOURCE = 'hand_tagged'

from src.utils import get_logger, load_env

def main():

    LOGGER = get_logger("add_hand_tag_journals")
    env_vars = load_env()


    print("\n----------------------------------------------------")
    LOGGER.info("Step 5: Improving field of study using hand-tagged journals")

    df = pd.read_csv(f'{env_vars["CSV_DATA_PATH"]}/labelled_journals.csv')
    df = df.dropna(subset=['field_of_study1'])
    df = df.fillna("")
    labelled_dicts = df.to_dict("records")
 
    # Connect to the SQLite database
    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    # get before count
    get_before_count_query = "SELECT COUNT(*) FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(get_before_count_query)
    before_count = cur.fetchone()[0]
    LOGGER.info(f"Number of citations with field of study before improving with hand-tagged journals: {before_count}")

    # select_venues_query = "SELECT DISTINCT journal FROM citations WHERE field_of_study IS NULL and year != -1 and authors != '' and id in (SELECT citation_id FROM article_citations WHERE section IS NOT NULL)"
    select_venues_query = "SELECT id, journal FROM citations WHERE journal != '' and (field_of_study IS NULL or fos_source = 'string_match')"

    cur.execute(select_venues_query)
    journals = cur.fetchall()

    for id, journal in journals:
        journal = journal.lower().strip()
        journal = " ".join(journal.split())
        for labelled_dict in labelled_dicts:
            labelled_journal = labelled_dict['journal'].lower().strip()
            labelled_journal = " ".join(labelled_journal.split())
            if journal == labelled_journal:
                field_of_study = str(labelled_dict['field_of_study1'])
                if labelled_dict['field_of_study2'] != "":
                    field_of_study += "; " + str(labelled_dict['field_of_study2'])
                field_of_study = field_of_study.replace(";", "; ")
                insert_query = f"UPDATE citations SET field_of_study = ? WHERE id = ?"
                cur.execute(insert_query, (field_of_study, id, ))
                insert_query = f"UPDATE citations SET fos_source = ? WHERE id = ?"
                cur.execute(insert_query, (SOURCE, id, ))
    
    # get after count
    get_after_count_query = "SELECT COUNT(*) FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(get_after_count_query)
    after_count = cur.fetchone()[0]
    LOGGER.info(f"Number of citations with field of study after improving with hand-tagged journals: {after_count}")

    con.commit()
    con.close()

if __name__ == "__main__":
    main()