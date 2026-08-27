import os
import sqlite3
import pickle
import pandas as pd


from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import clean_title, xmls_to_dict

con = sqlite3.connect(DB_FILENAME)
cur = con.cursor()

def get_year(row):
    cur.execute(f"SELECT year FROM article WHERE id = {row['article_id']};")
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

    add_one = "UPDATE article_citations SET count = 1 WHERE count = 0;"
    cur.execute(add_one)

    get_css = "SELECT article.year, citations.id, article_citations.count, citations.title, citations.meeting from article_citations \
                JOIN citations ON article_citations.citation_id = citations.id \
                    JOIN article ON article_citations.article_id = article.id\
                        WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND field_of_study IS NOT NULL and fos_source = 'semantic_scholar';"

    df = pd.read_sql_query(get_css, con)

    # print(df)
    df = df.groupby(['year', 'id']).agg({'title': 'max', 'meeting': 'max', 'count':'sum'}).reset_index()
    con.close()
    # df = df[df['year']==2014]
    # print(df.sort_values(by='count', ascending=False))
    # exit()

    years = df['year']
    titles = df['title']
    journals = df['meeting']
    counts = df['count']

    top_titles = {}
    for i, y in enumerate(years):
        if y not in top_titles:
            top_titles[y] = ["", 0]
        if counts[i] > top_titles[y][1]:
            if titles[i] != "Long Short-Term Memory":
                top_titles[y] = [titles[i], counts[i], journals[i]]

    for k, v in top_titles.items():
        print(f"Year: {k}, Title: {v[0]}, Count: {v[1]}")

if __name__ == "__main__":
    main()