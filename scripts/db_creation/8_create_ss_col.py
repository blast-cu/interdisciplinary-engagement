import sqlite3
import pandas as pd

from src.utils import get_logger, load_env

def main():
    LOGGER = get_logger("str_matching")
    env_vars = load_env()

    print("\n----------------------------------------------------")
    LOGGER.info("Step 8: creating social_science and interdisciplinary columns")
    
    # Connect to the SQLite database
    con = sqlite3.connect(env_vars['DB_PATH'])
    cur = con.cursor()

    # add column if it doesn't exist
    try:
        add_ss_col_query = "ALTER TABLE citations ADD COLUMN social_science BOOLEAN"
        cur.execute(add_ss_col_query)
    except sqlite3.OperationalError:
        LOGGER.debug("social_science column already exists")

    try:
        add_ss_col_query = "ALTER TABLE citations ADD COLUMN interdisciplinary BOOLEAN"
        cur.execute(add_ss_col_query)
    except sqlite3.OperationalError:
        LOGGER.debug("interdisciplinary column already exists")


    # get rows to iterate over
    fos_query = "SELECT id, field_of_study FROM citations WHERE field_of_study IS NOT NULL"
    cur.execute(fos_query)
    has_fos = cur.fetchall()

    f = open("data/csv_data/reviewer_labelled_fos.csv", "r")
    df = pd.read_csv(f)
    f.close()
    df = df.fillna(0)
    fields = df["field_of_study"].tolist()
    social_science = df["SS"].tolist()
    interdisciplinary = df["CD"].tolist()

    field_dict = {}
    for i in range(len(fields)):
        ss = int(social_science[i])
        cd = int(interdisciplinary[i])
        field_dict[fields[i]] = {}
        field_dict[fields[i]]["ss"] = ss
        field_dict[fields[i]]["cd"] = cd

    
    for row in has_fos:
        ss = 0
        inter = 0

        id = row[0]
        fos = row[1].lower().strip()
        if 'sciences' in fos:
            fos = fos.replace('sciences', 'science')
            update_query = f"UPDATE citations SET field_of_study = ? WHERE id = ?;"
            cur.execute(update_query, (fos, id, ))

        if ";" in fos:
            fos = fos.split(";")
        else:
            fos = [fos]
        
        for f in fos:
            f = f.strip()
            if f in field_dict:
                ss += field_dict[f]["ss"]
                inter += field_dict[f]["cd"]
            else:
                print(f)

        if ss > 0:
            ss = 1
        if inter > 0:
            inter = 1
        
        update_query = "UPDATE citations SET social_science = ? WHERE id = ?;"
        cur.execute(update_query, (ss, id,))

        update_query = "UPDATE citations SET interdisciplinary = ? WHERE id = ?;"
        cur.execute(update_query, (inter, id))

    # print number of social science and interdisciplinary citations
    get_ss_count_query = "SELECT COUNT(*) FROM citations WHERE social_science IS NOT NULL"
    cur.execute(get_ss_count_query)
    ss_count = cur.fetchone()[0]
    LOGGER.info(f"Number of citations with social_science value: {ss_count}")

    get_inter_count_query = "SELECT COUNT(*) FROM citations WHERE interdisciplinary IS NOT NULL"
    cur.execute(get_inter_count_query)
    inter_count = cur.fetchone()[0]
    LOGGER.info(f"Number of citations with interdisciplinary value: {inter_count}")

    con.commit()
    con.close()

if __name__ == "__main__":
    main()
