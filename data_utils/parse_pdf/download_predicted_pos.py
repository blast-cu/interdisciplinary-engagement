import sqlite3
import pandas as pd
import os
from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.collection.track_labelled_collector import download_paper_by_title
from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict
from data_utils.db_creation.utils import  get_local_article_path

# set up logging
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def make_dirs(venue, year):
    """
    create the directory structure we need for the venue
    """
    out_dir = f"{DATA_DIR}venues/{venue}/{year}/predicted_pos"
    raw_dir = f"{out_dir}/raw/"
    os.makedirs(raw_dir, exist_ok=True)

    processed_dir = f'{out_dir}/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    xml_out_dir = f"{out_dir}/xml/"
    os.makedirs(xml_out_dir, exist_ok=True)

    dict_out_dir = f"{out_dir}/dict/"
    os.makedirs(dict_out_dir, exist_ok=True)

    return raw_dir, xml_out_dir, processed_dir, dict_out_dir

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # read in the venue info
    venue_info_path = f"{DATA_DIR}csv_data/venue_info.csv"
    venue_info_df = pd.read_csv(venue_info_path)

    for index, row in venue_info_df.iterrows():
        # Process each row as needed
        venue, year, _, proceedings_url = row
        venue = venue.lower()

        # TEMPORARY, skip 2024 and 2025
        if year >= 2024:
            continue

        print(f"Processing row {venue} {year}")

        get_articles_query = f"SELECT article_id FROM predictions WHERE nlp_css_prediction = 1 AND needs_processing = 1 AND article_id IN (SELECT id FROM article WHERE venue = '{venue}' AND year = {year});"
        cur.execute(get_articles_query)
        article_ids = [a[0] for a in cur.fetchall()]

        downloaded_new = False
        for article_id in article_ids:

            # get other info from db\
            get_article_info_query = f"\
                SELECT title, pdf_id FROM article WHERE id = {article_id}\
            "
            cur.execute(get_article_info_query)
            title, pdf_id = cur.fetchone()

            # check that it isn't already there
            expected_path = get_local_article_path(venue, year, pdf_id, "dict", LOGGER, None)
            
            if expected_path is None:
                
                # make the directories, ok if they exist
                raw_dir, xml_out_dir, processed_dir, dict_out_dir = make_dirs(venue, year)
                print(f"Downloading {title}, pdf #{pdf_id} in {raw_dir}")

                # if not, download it
                download_paper_by_title(
                    proceedings_url, title, pdf_id, raw_dir, venue, year
                )
                downloaded_new = True

            # else it's already there!
            
        # if downloaded_new:
        #     generate_xml(
        #         pdf_path=raw_dir,
        #         out_dir=xml_out_dir,
        #         processed_dir=processed_dir
        #     )

        #     xmls_to_dict(xml_out_dir, dict_out_dir)


    con.close()





if __name__ == "__main__":
    main()