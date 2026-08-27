import sqlite3
import pandas as pd
from data_utils.utils import DB_FILENAME

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    get_args = "SELECT article_citations.arg_pred FROM article \
            JOIN article_citations ON article.id = article_citations.article_id \
                JOIN citations ON article_citations.citation_id = citations.id\
                    WHERE section IS NOT NULL AND  (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.interdisciplinary = 0;"
    
    df = pd.read_sql_query(get_args, con)
    df['count'] = 1
    df = df.groupby(['arg_pred']).count()
    df['count'] = df['count'] / df['count'].sum()
    df['percent'] = df['count'] * 100
    print("In discipline")
    print(df)
    print()

    get_args = "SELECT article_citations.arg_pred FROM article \
            JOIN article_citations ON article.id = article_citations.article_id \
                JOIN citations ON article_citations.citation_id = citations.id\
                    WHERE section IS NOT NULL AND  (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.interdisciplinary = 1;"
    
    df = pd.read_sql_query(get_args, con)
    df['count'] = 1
    df = df.groupby(['arg_pred']).count()
    df['count'] = df['count'] / df['count'].sum()
    df['percent'] = df['count'] * 100
    print("Out of discipline")
    print(df)
    print()

    get_args = "SELECT article_citations.arg_pred FROM article \
            JOIN article_citations ON article.id = article_citations.article_id \
                JOIN citations ON article_citations.citation_id = citations.id\
                    WHERE section IS NOT NULL AND  (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.social_science = 1;"
    
    df = pd.read_sql_query(get_args, con)
    df['count'] = 1
    df = df.groupby(['arg_pred']).count()
    df['count'] = df['count'] / df['count'].sum()
    df['percent'] = df['count'] * 100
    print("SS")
    print(df)
    con.close()

if __name__ == "__main__":
    main()