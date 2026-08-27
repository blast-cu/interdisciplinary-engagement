import os
import sqlite3
import pickle
import pandas as pd


from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import clean_title, xmls_to_dict

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    article_pos_query = "SELECT fos_source, COUNT(*) FROM citations WHERE field_of_study IS NOT NULL GROUP BY fos_source ORDER BY COUNT(*)"
    cur.execute(article_pos_query)
    articles = cur.fetchall()
    for a in articles:
        print(a)



    get_mh_query = "SELECT id, title, journal, meeting, field_of_study FROM citations WHERE field_of_study IS NOT NULL;"
    cur.execute(get_mh_query)
    mh_refs = cur.fetchall()
    for m in mh_refs:
        if 'mental health' in m[4]:
            print(m[0], m[1], m[2], m[3], m[4])

    con.close()





if __name__ == "__main__":
    main()