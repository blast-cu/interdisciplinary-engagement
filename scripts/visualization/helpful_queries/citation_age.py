import sqlite3
import matplotlib.style
import pandas as pd
import matplotlib.pyplot as plt

from data_utils.utils import DB_FILENAME

import matplotlib
from matplotlib import rcParams

def get_article_year(id, cur):
    select_query = f"SELECT year FROM article WHERE id = {id};"
    cur.execute(select_query)
    year = cur.fetchone()[0]
    return year

def get_article_year2(row, cur):
    select_query = f"SELECT year FROM article WHERE id = {row['id']};"
    cur.execute(select_query)
    year = cur.fetchone()[0]
    return year


def main():

    
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    matplotlib.style.use('seaborn-v0_8-darkgrid')


    select_query = "SELECT article_id, nlp_css_prediction FROM predictions;"
    cur.execute(select_query)

    for row in cur.fetchall():
        article_id = row[0]
        nlp_css_prediction = row[1]
        if nlp_css_prediction is not None:
            update_query = f"UPDATE article SET nlp_css_prediction = {nlp_css_prediction} WHERE id = {article_id};"
            # print(update_query)
            cur.execute(update_query)

    # create_ml_col = "ALTER TABLE citations ADD COLUMN ml INTEGER;"
    # cur.execute(create_ml_col)
    update_count = "UPDATE article_citations SET count = 1 WHERE count = 0;"

    select_query = "SELECT article.id, citations.year, citations.interdisciplinary FROM article \
                        JOIN article_citations ON article.id = article_citations.article_id \
                            JOIN citations ON article_citations.citation_id = citations.id\
                                WHERE article.year != -1 AND citations.year != -1 AND citations.interdisciplinary IS NOT NULL AND citations.year IS NOT NULL AND (article.nlp_css = 1 OR article.nlp_css_prediction = 1);"
    df = pd.read_sql_query(select_query, con)
    df['article_year'] = df['id'].map(lambda x: get_article_year(x, cur))
    df['age'] =  df['article_year'] - df['year']
    

    df = df.drop(columns=['article_year', 'year'])
    df = df.groupby(['id', 'interdisciplinary']).mean()
    df['average age'] = df['age']
    df = df.drop(columns=['age'])
    df = df.unstack()
    df['year'] = df.index.map(lambda x: get_article_year(x, cur))
    df.groupby(['year',]).mean().plot()
    print(df)
    plt.legend(['In Domain', 'Out-of-Domain'])
    plt.title('Average Citation Age in a Publication by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Citation Age')
    plt.tight_layout()
    plt.savefig('citation_age.png')
    plt.show()


    con.close()

if __name__ == "__main__":
    main()