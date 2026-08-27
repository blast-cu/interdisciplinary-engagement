import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from data_utils.utils import DB_FILENAME

def get_year(id, cur):
    select_query = f"SELECT year FROM article WHERE id = {id};"
    cur.execute(select_query)
    year = cur.fetchone()[0]
    return year

def get_ml(row):
    #  NeurIPs/NIPS, ICLR, ICML, AISTATS, COLT, JMLR, TMLR, AAAI, IJCAI, JAIR, UAI
    meeting = row['meeting'].lower()
    journal = row['journal'].lower()
    ml = 0
    if 'machine learning' in meeting or 'machine learning' in journal:
        ml = 1
    elif 'artificial intelligence' in meeting or 'artificial intelligence' in journal:
        ml = 1
    elif 'neural' in meeting or 'neural' in journal:
        ml = 1
    elif 'nips' in meeting or 'nips' in journal:
        ml = 1
    elif 'neurips' in meeting or 'neurips' in journal:
        ml = 1
    elif 'iclr' in meeting or 'iclr' in journal:
        ml = 1
    elif 'icml' in meeting or 'icml' in journal:
        ml = 1
    elif 'aistats' in meeting or 'aistats' in journal:
        ml = 1
    elif 'colt' in meeting or 'colt' in journal:
        ml = 1
    elif 'jmlr' in meeting or 'jmlr' in journal:
        ml = 1
    elif 'tmlr' in meeting or 'tmlr' in journal:
        ml = 1
    elif 'aaai' in meeting or 'aaai' in journal:
        ml = 1
    elif 'ijcai' in meeting or 'ijcai' in journal:
        ml = 1
    elif 'jair' in meeting or 'jair' in journal:
        ml = 1
    elif 'uai' in meeting or 'uai' in journal:
        ml = 1
    return ml

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

    # create_ml_col = "ALTER TABLE citations ADD COLUMN ml INTEGER;"
    # cur.execute(create_ml_col)
    update_count = "UPDATE article_citations SET count = 1 WHERE count = 0;"

    select_query = "SELECT article.id, article_citations.section, citations.journal, citations.meeting, article_citations.count FROM article \
                        JOIN article_citations ON article.id = article_citations.article_id \
                            JOIN citations ON article_citations.citation_id = citations.id\
                                WHERE citations.meeting IS NOT NULL or citations.journal IS NOT NULL AND citations.interdisciplinary = 0 AND (article.nlp_css = 1 OR article.nlp_css_prediction = 1);"
    df = pd.read_sql_query(select_query, con)
    df = df.drop(columns=['section'])

    df['ml'] = df.apply(lambda x: get_ml(x), axis=1)
    # df['count'] = 1

    # print(df.sort_values(by='ml', ascending=False))
    df = df.groupby(['id']).sum()
    df['ml'] = df['ml'] / df['count']
    df = df.drop(columns=['count'])
    df['year'] = df.index.map(lambda x: get_year(x, cur))
    print(df)
    df = df.boxplot(column='ml', by='year')
    

    df.plot()
    plt.title('Average Proportion of Machine Learning Citations in a Publication by Year')
    plt.suptitle('')
    plt.xlabel('Year')
    plt.ylabel('Proportion of Machine Learning Citations')
    plt.tight_layout()
    plt.savefig('ml_citations.png')
    plt.show()


    con.close()

if __name__ == "__main__":
    main()