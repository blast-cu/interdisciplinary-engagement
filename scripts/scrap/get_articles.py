import pandas as pd
import sqlite3


citation_path = "/Users/alexleto/projects/cross_disciplinary_engagement/data/manual_anns/citations.csv"
c_df = pd.read_csv(citation_path)
# article_ids = c_df["article_id"].tolist()

con = sqlite3.connect("data/data.db")
cur = con.cursor()

a_df = c_df[['article_id']]
a_df = a_df.drop_duplicates(subset=["article_id"])

def get_year(id): 
    cur.execute(f"SELECT year FROM article WHERE id = '{id}';")
    return cur.fetchone()[0]

def get_venue(id):
    cur.execute(f"SELECT venue FROM article WHERE id = '{id}';")
    return cur.fetchone()[0]

a_df["year"] = a_df["article_id"].apply(get_year)
a_df["venue"] = a_df["article_id"].apply(get_venue)
a_df.to_csv("articles.csv")


