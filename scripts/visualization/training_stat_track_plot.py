import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import sqlite3
import os
import pandas as pd


from data_utils.utils import DB_FILENAME
from data_utils.parse_pdf.utils import clean_title


def main():

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # add_ss_col_query = "ALTER TABLE article ADD COLUMN track TEXT;"
    # cur.execute(add_ss_col_query)

    df = pd.read_csv('data/css_track_papers.csv')
    tracks = df.to_dict(orient='records')
    for entry in tracks:
        entry_title = clean_title(entry['title'].strip())
        entry_venue = entry['conference']
        entry_year = entry['year']
        entry_track = entry['track']
        found_it = False

        select_entry_query = f"SELECT id, title FROM article WHERE venue = '{entry_venue}' AND year = {entry_year};"
        # print(select_entry_query)
        
        cur.execute(select_entry_query)
        rows = cur.fetchall()
        for row in rows:
            row_id = row[0]
            row_title = clean_title(row[1].strip())
            # print(row_title)
            
            if entry_title in row_title:
                # print(f"Found {entry_title} in the database")
                entry_track = entry_track.lower().strip()
                entry_track = entry_track.replace(", and", " and")
                update_query = f"UPDATE article SET track = '{entry_track.lower().strip()}' WHERE id = {row_id};"
                # print(update_query)
                cur.execute(update_query)
                found_it = True
                break
        if not found_it:
            print(f"Could not find {entry_title} in the database")
            con.close()
            exit()

    get_tracks = "SELECT track, COUNT(*) FROM article GROUP BY track;"
    df = pd.read_sql(get_tracks, con).set_index('track')
    df.to_csv('data_utils/visualization/output/tracks.csv')
    con.commit()
    con.close()

    exit()

    output_dir = "data_utils/visualization/output"



    # print(matplotlib.style.available)






if __name__ == "__main__":
    main()