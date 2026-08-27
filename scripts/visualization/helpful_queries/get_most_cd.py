import os
import sqlite3
import pickle
import pandas as pd


from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import clean_title, xmls_to_dict

con = sqlite3.connect(DB_FILENAME)
cur = con.cursor()

def get_title(x, cur):
    cur.execute(f"SELECT title FROM article WHERE id = {x};")
    return cur.fetchone()[0]

def main():

    select_query = "SELECT article_id, nlp_css_prediction FROM predictions;"
    cur.execute(select_query)

    for row in cur.fetchall():
        article_id = row[0]
        nlp_css_prediction = row[1]
        if nlp_css_prediction is not None:
            update_query = f"UPDATE article SET nlp_css_prediction = {nlp_css_prediction} WHERE id = {article_id};"
            # print(update_query)
            cur.execute(update_query)
    
    most_cd_query = "SELECT article.id, citations.interdisciplinary from article\
                        JOIN article_citations ON article.id = article_citations.article_id\
                            JOIN citations ON article_citations.citation_id = citations.id\
                                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND field_of_study IS NOT NULL;"
    df = pd.read_sql_query(most_cd_query, con)
    df['count'] = 1
    df = df.groupby('id').sum()
    df['percent interdisciplinary'] = df['interdisciplinary'] / df['count']


    df = df.sort_values(by='percent interdisciplinary', ascending=False)
    df['title'] = df.index.map(lambda x: get_title(x, cur))
    df.head().to_csv("scratch/most_cd.csv")

    df = df.sort_values(by='percent interdisciplinary', ascending=True)
    df.to_csv("scratch/least_cd.csv")


if __name__ == "__main__":
    main()