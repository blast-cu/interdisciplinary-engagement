import sqlite3
import pandas as pd

from src.utils import load_env, get_logger
from src.db_creation import clean_string

def main():

    LOGGER = get_logger("str_matching")
    env_vars = load_env()

    print("\n----------------------------------------------------")
    LOGGER.info("Step 7: improving field of study using string matching")

    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    SOURCE = 'string_match'

    # get before count
    select_venues_query = "SELECT id, journal FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(select_venues_query)
    existing_fos = cur.fetchall()
    existing_fos_count = str(len(existing_fos))
    LOGGER.info(f"Number of citations with field of study before improving with string matching: {existing_fos_count}")

    get_nones_query = "SELECT id, journal, meeting FROM citations \
        WHERE (fos_source = 'string_match' OR field_of_study IS NULL) AND (journal IS NOT NULL OR meeting IS NOT NULL)"
    cur.execute(get_nones_query)

    none_refs = cur.fetchall()
    before_count = len(none_refs)

    df = pd.read_csv(f"{env_vars['CSV_DATA_PATH']}/reviewer_labelled_fos.csv")
    fields = df['field_of_study'].tolist()
    fields = [v.lower().strip() for v in fields]
    fields = list(set(fields))
    
    fos_dict = {
        "liguistic": "linguistics",
        "economic": "economics",
        "biological": "biology",
        "ethic": "ethics",
        "genetic": "genetics",
        "geographic": "geography",
        "historical": "history",
        "medical": "medicine"
    }
    for field in fields:
        if "(" in field:
            field_edit = field.split("(")[0].strip()
            fos_dict[field_edit] = field
        else:
            fos_dict[field] = field


    for ref in none_refs:
        id = ref[0]
        to_check = []
        journal = ref[1]
        meeting = ref[2]
        if journal is not None and journal != '':
            to_check.append(journal.lower().strip())
        if meeting is not None and meeting != '':
            to_check.append(meeting.lower().strip())

        matched_fields = []
        for paper_venue in to_check:  # check both journal and meeting
            for fos in fos_dict.keys():
                if fos.lower() in paper_venue:
                    matched_fields.append(fos_dict[fos])
        if len(matched_fields) > 0:
            field_of_study = matched_fields[0]
            if len(matched_fields) > 1:
                matched_fields = list(set(matched_fields))
                field_of_study = ('; ').join(matched_fields)
            field_of_study = clean_string(field_of_study)
            update_query = f"UPDATE citations SET field_of_study = ? WHERE id = ?;"
            cur.execute(update_query, (field_of_study, id,))
            update_query = f"UPDATE citations SET fos_source= ? WHERE id = ?;"
            cur.execute(update_query, (SOURCE, id,))

    # get after count
    select_venues_query = "SELECT id, journal FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(select_venues_query)
    after_fos = cur.fetchall()
    after_fos_count = str(len(after_fos))
    LOGGER.info(f"Number of citations with field of study after improving with string matching: {after_fos_count}")

    con.commit()
    con.close()
    

if __name__ == "__main__":
    main()