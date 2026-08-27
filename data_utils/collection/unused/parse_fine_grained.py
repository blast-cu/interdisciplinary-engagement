import argparse
import sqlite3
import os

from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict


def main():  # come back to EACL 2023, NAACL 2022

    venue = 'fine_grain'
    year = '0000'

    out_dir = f"{DATA_DIR}venues/{venue}/{year}/predicted_pos"
    raw_dir = f"{out_dir}/raw/"
    os.makedirs(raw_dir, exist_ok=True)

    processed_dir = f'{out_dir}/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    xml_out_dir = f"{out_dir}/xml/"
    os.makedirs(xml_out_dir, exist_ok=True)

    dict_out_dir = f"{out_dir}/dict/"
    os.makedirs(dict_out_dir, exist_ok=True)

    generate_xml(
        pdf_path=raw_dir,
        out_dir=xml_out_dir,
        processed_dir=processed_dir
    )

    xmls_to_dict(xml_out_dir, dict_out_dir)
    



if __name__ == "__main__":
    main()