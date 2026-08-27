import os
import sqlite3
import pickle
import pandas as pd
import requests
import json
from habanero import Crossref

from data_utils.utils import DB_FILENAME
from data_utils.parse_pdf.utils import clean_title

def update_fos(journal, fos_list, id, cur, to_update):
    journal = journal.lower()
    journal = " ".join(journal.split())
    journal_fields = []

    for fos in fos_list:
        if fos in journal:
            journal_fields.append(fos)
    
    if journal_fields != []:
        if id not in to_update:
            to_update[id] = []
        to_update[id] += journal_fields
    
    return to_update

def main():

    df = pd.read_csv("data/labelled_fos.csv")
    fos_list = df["field_of_study"].tolist()
    fos_list = [f.lower() for f in fos_list]

    # Connect to the SQLite database
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    # select_venues_query = "SELECT DISTINCT journal FROM citations WHERE field_of_study IS NULL and year != -1 and authors != '' and id in (SELECT citation_id FROM article_citations WHERE section IS NOT NULL)"
    select_venues_query = "SELECT id, journal FROM citations WHERE journal != '' and (field_of_study IS NULL or field_of_study = '' or fos_source = 'HABANERO')"
    # select_venues_query = "SELECT id, journal FROM citations WHERE journal != '' and (fos_source = 'fos_str_match')"
    cur.execute(select_venues_query)
    journals = cur.fetchall()
    to_update = {}
    for id, journal in journals:
        to_update = update_fos(journal, fos_list, id, cur, to_update)
        
    select_venues_query = "SELECT id, meeting FROM citations WHERE meeting != '' and (field_of_study IS NULL or fos_source = 'HABANERO')"
    # select_venues_query = "SELECT id, journal FROM citations WHERE journal != '' and (fos_source = 'fos_str_match')"
    cur.execute(select_venues_query)
    meetings = cur.fetchall()
    for id, meeting in meetings:
        to_update = update_fos(meeting, fos_list, id, cur, to_update)

    for id, fields in to_update.items():
        fields = list(set(fields))
        journal_fields = "; ".join(fields)
        update_query = f"UPDATE citations SET field_of_study = '{journal_fields}' WHERE id = {id}"
        print(update_query)
        cur.execute(update_query)
        update_query = f"UPDATE citations SET fos_source = 'fos_str_match' WHERE id = {id}"
        print(update_query)
        print()
        cur.execute(update_query)
    
    con.commit()
    con.close()

    # journal_dict = {}
    # journal_dict['journal'] = []

    # # Connect to the SQLite database
    # con = sqlite3.connect(DB_FILENAME)
    # cur = con.cursor()
    # select_venues_query = "SELECT DISTINCT journal FROM citations WHERE field_of_study IS NULL and year != -1 and authors != '' and id in (SELECT citation_id FROM article_citations WHERE section IS NOT NULL)"

    # cur.execute(select_venues_query)
    # journals = cur.fetchall()
    # select_meetings_query = "SELECT DISTINCT meeting FROM citations WHERE field_of_study IS NULL and year != -1 and authors != ''"
    # cur.execute(select_meetings_query)
    # meetings = cur.fetchall()
    # print(len(meetings))

    # journals.extend(meetings)

    # journals = [j[0].lower().strip() for j in journals]
    # journals = [j for j in journals if j != '']
    # journals = [" ".join(j.split()) for j in journals]

    # journals_clean = []
    # for j in journals:
    #     keyword_match = False
    #     for fos in fos_list:
    #         if fos in j:
    #             keyword_match = True
    #             break
    #     if not keyword_match:
    #         journals_clean.append(j)

    # journals = list(set(journals_clean))
    # journal_dict['journal'] = journals
    # print(len(journals))
    # con.close()

    # pd.DataFrame(journal_dict).to_csv("journals_to_label.csv", index=False)





if __name__ == "__main__":
    main()