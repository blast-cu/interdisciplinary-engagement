import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm


from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict
from data_utils.db_creation.utils import clean_title, create_connection
import logging

import sqlite3

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

VENUE_INFO_FILE = f"{DATA_DIR}venue_info.csv"
TRACK_FILE = f'{DATA_DIR}css_track_papers.csv'


def main():

    # read in venue info 
    venue_df = pd.read_csv(VENUE_INFO_FILE)
    track_df = pd.read_csv(TRACK_FILE)

    old_db_path = f"{DATA_DIR}data.db"

    for row in venue_df.itertuples():
        for NLP_CSS_LABEL in ["pos", "neg"]:

            # read conference info off csv
            YEAR = row.year
            CONFERENCE = row.venue
            URL = row.proceedings_url

            # create output directories for processing
            OUT_DIR, raw_dir, processed_dir, xml_out_dir, dict_out_dir = \
                create_out_dirs(DATA_DIR, CONFERENCE, YEAR, NLP_CSS_LABEL)

            filtered_df = track_df[track_df['conference'] == CONFERENCE]
            filtered_df = filtered_df[filtered_df['year'] == YEAR]

            # filter to positive or negative labels
            nlp_css_val = 0 
            if NLP_CSS_LABEL == "pos":
                nlp_css_val = 1
            filtered_df = filtered_df[filtered_df['nlp_css'] == nlp_css_val]
            
            titles = filtered_df['title'].values
            titles = [t.strip() for t in titles]

            if len(titles) == 0:
                LOGGER.warning(f"No papers found in track db for {CONFERENCE} {YEAR}, skipping...")
                continue

            for id, title in tqdm(enumerate(titles), desc=f"Downloading papers from {CONFERENCE} {YEAR}"):
                # check if paper is already in (old or new) db
                # if it is, update nlp_css label
                in_db = False
                if check_title_in_db(DB_FILENAME, title, CONFERENCE, YEAR):
                    # update nlp_css label in db
                    update_nlp_css_label(DB_FILENAME, title, CONFERENCE, YEAR, nlp_css_val)
                    in_db = True
                
                if check_title_in_db(OLD_DB_FILENAME, title, CONFERENCE, YEAR):
                    # update nlp_css label in db
                    update_nlp_css_label(OLD_DB_FILENAME, title, CONFERENCE, YEAR, nlp_css_val)
                    in_db = True

                if not in_db:
                    # download paper
                    download_paper_by_title(URL, title, id, raw_dir, CONFERENCE, YEAR)

    
if __name__ == "__main__":
    main()