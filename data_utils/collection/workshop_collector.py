import os
import pandas as pd
import logging

from data_utils.utils import DATA_DIR
from data_utils.collection.utils import download_papers
from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def main():
    """
    Downloads all the pdfs for workshops listed in data/csv_data/workshop_info.csv
    """

    # read in workshop data
    workshop_df = pd.read_csv(f"{DATA_DIR}/csv_data/workshop_info.csv")

    for row in workshop_df.itertuples():
        # read 
        YEAR = row.year
        WORKSHOP_NAME = row.workshop
        URL = row.url

        LOGGER.info(f"processing {WORKSHOP_NAME} {YEAR}")

        OUT_DIR = f"{DATA_DIR}venues/{WORKSHOP_NAME}/{YEAR}/"
        os.makedirs(OUT_DIR, exist_ok=True)

        raw_out_dir = OUT_DIR + "raw/"
        os.makedirs(raw_out_dir, exist_ok=True)

        xml_out_dir = OUT_DIR + "xml/"
        os.makedirs(xml_out_dir, exist_ok=True)

        processed_dir = OUT_DIR + "processed/"
        os.makedirs(processed_dir, exist_ok=True)

        dict_out_dir = OUT_DIR + "dict/"
        os.makedirs(dict_out_dir, exist_ok=True)

        # download all the workshop papers
        download_papers(
            URL,
            raw_out_dir,
            download=True
        )

    
if __name__ == "__main__":
    main()
