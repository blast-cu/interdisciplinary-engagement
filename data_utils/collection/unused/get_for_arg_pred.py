import pickle
import pandas as pd
import re
import sqlite3
from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import xmls_to_dict
import json

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    # get all rows with no section
    to_predict = {}
    get_no_section_query = "SELECT * FROM article_citations WHERE section IS NOT NULL"
    cur.execute(get_no_section_query)
    print(cur.description)
    for row in cur.fetchall():
        id = str(row[0]) + "_" + str(row[1])  # article_id_citation_id
        to_predict[id] = row[4]
    print(len(to_predict))

    pickle.dump(to_predict, open(f"{DATA_DIR}to_arg_predict.pkl", "wb"))

    # j = json.load(open(f"{DATA_DIR}arg_predictions.json", "rb"))
    # print(len(j))
if __name__ == "__main__":
    main()