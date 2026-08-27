
from data_utils.utils import DATA_DIR, DB_FILENAME
import sqlite3
import os

def main():
    venue = 'ACL'
    venue_years = [2014, 2015, 2017]

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    for year in venue_years:
        print(f"Venue: {venue}, Year: {year}")
        predicted_pos_query = f"SELECT article_id FROM predictions WHERE nlp_css = 1 AND article_id IN\
                                    (SELECT id from article WHERE venue = '{venue}' AND year = {year})"
        cur.execute(predicted_pos_query)
        predicted_pos = cur.fetchall()

        pos_xml_dir = f"{DATA_DIR}venues/{venue}/{year}/unlabelled/xml/"
        predicted_pos_xml_dir = f"{DATA_DIR}venues/{venue}/{year}/predicted_pos/xml/"
        predicted_pos_dict_dir = f"{DATA_DIR}venues/{venue}/{year}/predicted_pos/dict/"
        os.makedirs(predicted_pos_dict_dir, exist_ok=True)
        os.makedirs(predicted_pos_xml_dir, exist_ok=True)

        # for id in predicted_pos:
        #     # get pdf id
        #     pdf_id_query = f"SELECT pdf_id FROM article WHERE id = {id[0]}"
        #     cur.execute(pdf_id_query)
        #     pdf_id = cur.fetchone()[0]

        #     # move xml file
        #     xml_file = f"{pdf_id}.grobid.tei.xml"
        #     os.rename(f"{pos_xml_dir}{xml_file}", f"{predicted_pos_xml_dir}{xml_file}")


    con.close()

if __name__ == "__main__":
    main()