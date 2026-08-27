import os
import sqlite3
import pickle
import pandas as pd
import re

from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import xmls_to_dict

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # create table for in text citation prediction
    create_table = "CREATE TABLE IF NOT EXISTS in_text_citation (article_id INTEGER PRIMARY KEY, prediction TEXT);"

    # loop over each conference/year
    df = pd.read_csv(f"{DATA_DIR}venue_info.csv")
    for ind in df.index:

        venue_year = df['year'][ind]
        venue = df['venue'][ind]

        if venue != 'NAACL':  # only do NAACL for now
            continue

        print(f"Venue: {venue}, Year: {venue_year}")
        for cat in ['predicted_pos', 'pos']:
            pickle_dir = f"{DATA_DIR}venues/{venue}/{venue_year}/{cat}/dict/"
            xml_dir = f"{DATA_DIR}venues/{venue}/{venue_year}/{cat}/xml/"

            if not os.path.exists(pickle_dir):
                if cat == 'pos':
                    print(f"Directory {pickle_dir} does not exist")
                    continue
                else:  # CHANGE LATER
                    print(f"Directory {pickle_dir} does not exist")
                    continue

            xmls_to_dict(xml_dir, pickle_dir)