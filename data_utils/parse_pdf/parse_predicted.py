import argparse
import sqlite3
import os

from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import generate_xml, xmls_to_dict


def main(args):  # come back to EACL 2023, NAACL 2022
    year = args.y
    venue = args.v

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    get_articles_query = f"SELECT article_id FROM predictions WHERE nlp_css_prediction = 1 AND needs_processing = 1 AND article_id IN (SELECT id FROM article WHERE venue = '{venue}' AND year = {year});"
    cur.execute(get_articles_query)
    articles = cur.fetchall()
    

    print("\n -------------------")
    print(f"Parsing {venue} {year} papers")
    print(" -------------------")
    print(f"Number of articles: {len(articles)}")

    old_dir = f"{DATA_DIR}venues/{venue}/{year}/unlabelled/raw/"

    out_dir = f"{DATA_DIR}venues/{venue}/{year}/predicted_pos"
    raw_dir = f"{out_dir}/raw/"
    os.makedirs(raw_dir, exist_ok=True)

    processed_dir = f'{out_dir}/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    xml_out_dir = f"{out_dir}/xml/"
    os.makedirs(xml_out_dir, exist_ok=True)

    dict_out_dir = f"{out_dir}/dict/"
    os.makedirs(dict_out_dir, exist_ok=True)

    # move pdf files to raw directory
    for id in articles:
        id = id[0]
        pdf_id_query = f"SELECT pdf_id FROM article WHERE id = {id}"
        cur.execute(pdf_id_query)
        pdf_id = cur.fetchone()[0]


        old_pdf_path = f"{old_dir}{pdf_id}.pdf"
        new_pdf_path = f"{raw_dir}{pdf_id}.pdf"

        if os.path.isfile(old_pdf_path):
            os.rename(old_pdf_path, new_pdf_path)
        elif os.path.isfile(new_pdf_path):
            continue
        elif os.path.isfile(f"{processed_dir}{pdf_id}.pdf"):
            continue
        else:
            # raise Exception(f"File {pdf_id}.pdf not found")
            print(f"File {pdf_id}.pdf not found")
            continue

    con.close()

    generate_xml(
        pdf_path=raw_dir,
        out_dir=xml_out_dir,
        processed_dir=processed_dir
    )

    xmls_to_dict(xml_out_dir, dict_out_dir)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--v', required=True, help='venue')
    parser.add_argument('--y', required=False, help='year')
    args = parser.parse_args()
    main(args)