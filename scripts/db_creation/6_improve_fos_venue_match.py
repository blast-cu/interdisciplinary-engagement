import os
import sqlite3
import pickle
import pandas as pd

from src.utils import load_env, get_logger
from src.db_creation import clean_string
SOURCE = 'hand_tagged'

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def main():
    LOGGER = get_logger("add_hand_tag_venues")
    env_vars = load_env()

    print("\n----------------------------------------------------")
    LOGGER.info("Step 6: Improving field of study using hand tagged ss venues")

    # Connect to the SQLite database
    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    # get before count
    select_venues_query = "SELECT id, journal FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(select_venues_query)
    existing_fos = cur.fetchall()
    existing_fos_count = len(existing_fos)
    LOGGER.info(f"Number of citations with field of study before improving with ss venues: {existing_fos_count}")

    get_nones_query = "SELECT id, journal, meeting FROM citations \
        WHERE (fos_source = 'str_match' OR field_of_study IS NULL) AND (journal IS NOT NULL OR meeting IS NOT NULL)"
    cur.execute(get_nones_query)

    none_refs = cur.fetchall()
    before_count = len(none_refs)

    df = pd.read_csv(f"{env_vars['CSV_DATA_PATH']}/ss_venues.csv")
    venues = df['venue'].tolist()
    venues = [v.lower().strip() for v in venues]
    field = df['field of study'].tolist()
    ven_dict = dict(zip(venues, field))
    
    for ref in none_refs:
        id = ref[0]
        to_check = []
        journal = ref[1]
        meeting = ref[2]
        if journal is not None and journal != '':
            to_check.append(journal.lower().strip())
        if meeting is not None and meeting != '':
            to_check.append(meeting.lower().strip())

        for paper_venue in to_check:  # check both journal and meeting

            if paper_venue in ven_dict.keys():
                field_of_study = ven_dict[paper_venue.lower().strip()]
                field_of_study = clean_string(field_of_study)
                update_query = f"UPDATE citations SET field_of_study = ? WHERE id = ?"
                cur.execute(update_query, (field_of_study, id))
                update_query = f"UPDATE citations SET fos_source= ? WHERE id = ?"
                cur.execute(update_query, (SOURCE, id, ))
                break

    get_nones_query = "SELECT id, journal, meeting FROM citations \
        WHERE field_of_study IS NULL AND (journal IS NOT NULL OR meeting IS NOT NULL)"
    cur.execute(get_nones_query)
    after_count = len(cur.fetchall())
    # diff = before_count - after_count
    # print("Added field of study to", diff, "citations")
    LOGGER.info(f"Number of citations with field of study after improving with ss venues: {existing_fos_count + (before_count - after_count)}")

    fields = set()
    get_fos_query = "SELECT field_of_study FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(get_fos_query)
    fos = cur.fetchall()

    for f in fos:
        # print(f)
        current = f[0].split('; ')
        for c in current:
            fields.add(c.lower())
    # print("Fields of study:", fields)

    get_fos_source_query = "SELECT fos_source, count(*) FROM citations WHERE fos_source IS NOT NULL GROUP BY fos_source"
    cur.execute(get_fos_source_query)
    sources = str(cur.fetchall())
    LOGGER.info(f"Sources of field of study: {sources}")

    con.commit()
    con.close()

if __name__ == "__main__":
    main()
