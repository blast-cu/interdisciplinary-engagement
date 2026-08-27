import sqlite3
import pandas as pd
from data_utils.utils import DB_FILENAME

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    
    # print all tables in the database
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print("Tables in the database:")
    for table in tables:
        print(f" - {table[0]}")
    print()

    # summarize article table
    print("Article Table Summary:")
    select_query = "SELECT year, venue FROM article WHERE nlp_css = 1;"
    df_articles = pd.read_sql_query(select_query, con)
    article_summary = df_articles.groupby(['year']).size().reset_index(name='count')
    print(article_summary)
    print(f"Total NLP+CSS Articles: {len(df_articles)}\n")

    # print venues
    venue_counts = df_articles['venue'].value_counts()
    print("Venue Counts:")
    print(venue_counts)
    print()


    print("Citation Table Summary:")
    select_query = f"SELECT * FROM article_citations WHERE article_id IN (SELECT id FROM article WHERE nlp_css = 1);"
    article_citations = pd.read_sql_query(select_query, con)
    print(f"Total Citations: {len(article_citations)}")

    # stats on citation per article: min/max/avg/median
    citation_counts = article_citations.groupby('article_id').size()
    print(f"Min Citations per Article: {citation_counts.min()}")
    print(f"Max Citations per Article: {citation_counts.max()}")
    print(f"Avg Citations per Article: {citation_counts.mean():.2f}")
    print(f"Median Citations per Article: {citation_counts.median()}")

    # close the connection
    con.close()


    

if __name__ == "__main__":
    main()