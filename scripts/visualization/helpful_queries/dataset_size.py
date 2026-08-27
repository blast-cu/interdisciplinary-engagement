import sqlite3
import pandas as pd
from data_utils.utils import DB_FILENAME

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    select_query = "SELECT article_id, nlp_css_prediction FROM predictions;"
    cur.execute(select_query)

    for row in cur.fetchall():
        article_id = row[0]
        nlp_css_prediction = row[1]
        if nlp_css_prediction is not None:
            update_query = f"UPDATE article SET nlp_css_prediction = {nlp_css_prediction} WHERE id = {article_id};"
            # print(update_query)
            cur.execute(update_query)

    get_articles_query = "SELECT count(*) FROM article WHERE (nlp_css = 1 or nlp_css_prediction = 1);"
    print(cur.execute(get_articles_query).fetchall())


    get_sec = "SELECT count(*) FROM article \
                JOIN article_citations ON article.id = article_citations.article_id\
                    WHERE (article.nlp_css = 1 or article.nlp_css_prediction = 1) AND article_citations.section is NOT NULL ;"
    print(cur.execute(get_sec).fetchall())

    con.close()

    # list ss 
    df = pd.read_csv("data/hand_tagged/labelled_fos.csv")
    df = df[df['SS']==1]
    fos_list = df['field_of_study'].unique().tolist()
    fos_list = (", ").join(fos_list)
    print(fos_list) 


if __name__ == "__main__":
    main()