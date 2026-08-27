import sqlite3
from src.utils import load_env


def main():
    env_vars = load_env()
    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    # get all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    # get row counts for each table
    print("Tables in the database:")
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table[0]};")
        count = cur.fetchone()[0]
        print(f"- {table[0]}: {count} rows")


    # label summary
    print("\nTrack Label Summary: ")
    select_labels_query = "SELECT nlp_css, COUNT(id) FROM article GROUP BY nlp_css"
    cur.execute(select_labels_query)
    print(cur.fetchall())

    # check article citations
    print("\nCitations [TODO - check this]: ")
    select = "SELECT a.year, COUNT(ac.article_id)\
        FROM article as a RIGHT JOIN article_citations_context as ac ON a.id = ac.article_id\
        GROUP BY a.year;"
    cur.execute(select)
    for year, count in cur.fetchall():
        print(f"{year}: {count} in-text citations")


    # get citations stats
    print("\nGetting social science stats")
    get_ood_query = "SELECT citations.interdisciplinary, COUNT(article_citations.article_id) FROM article \
        JOIN article_citations ON article.id = article_citations.article_id \
            JOIN citations ON article_citations.citation_id = citations.id \
                WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1) AND citations.field_of_study IS NOT NULL \
                    GROUP BY citations.interdisciplinary;"
    cur.execute(get_ood_query)         
    print(cur.fetchall())

    # Number of papers
    # Number of citations
    # Average number of interdisciplinary citations per paper

    # get number of papers 
    select_pos = "SELECT COUNT(*) FROM article WHERE (article.nlp_css = 1 OR article.nlp_css_prediction = 1);"
    cur.execute(select_pos)
    print(f"\nNumber of NLP+CSS articles in db: {cur.fetchone()[0]}")

    # get number of citations
    select_in_text = "SELECT COUNT(*) FROM article as a \
          RIGHT JOIN article_citations as ac ON a.id = ac.article_id \
            LEFT JOIN citations as c ON ac.citation_id = c.id\
                WHERE (a.nlp_css = 1 OR a.nlp_css_prediction = 1) AND (c.field_of_study IS NOT NULL);"
    cur.execute(select_in_text)
    print(f"Number of citations w fos: {cur.fetchone()[0]}")


    # get number of cited works 
    select_cited = "SELECT COUNT(*) FROM citations as c\
        WHERE id IN \
            (SELECT DISTINCT ac.citation_id from article_citations_context as ac JOIN article as a on ac.article_id = a.id \
                WHERE (a.nlp_css = 1 OR a.nlp_css_prediction = 1))\
                AND (c.field_of_study IS NOT NULL)"
    cur.execute(select_cited)
    cited_work_count = cur.fetchone()[0]

    select_it_cited = "\
            SELECT COUNT(acc.citation_id) \
                FROM citations as c \
                    JOIN article_citations_context as acc ON c.id = acc.citation_id \
                    JOIN article as a ON acc.article_id = a.id \
                WHERE (a.nlp_css = 1 OR a.nlp_css_prediction = 1) AND (c.field_of_study IS NOT NULL) AND (acc.paragraph_context IS NOT NULL);\
            "
    cur.execute(select_it_cited)
    cited_it_count = cur.fetchone()[0]
    print(f"Number of cited works w fos {cited_work_count} ({cited_it_count} in-text)")

    # get number of interdisciplinary citations
    select_in_text = "SELECT COUNT(*) FROM article as a \
          RIGHT JOIN article_citations as ac ON a.id = ac.article_id \
            LEFT JOIN citations as c ON ac.citation_id = c.id\
                WHERE (a.nlp_css = 1 OR a.nlp_css_prediction = 1) AND (c.field_of_study IS NOT NULL) AND (c.interdisciplinary = 1);"
    cur.execute(select_in_text)
    interdisciplinary_count = cur.fetchone()[0]

    select_it_in_cited = "\
            SELECT COUNT(acc.citation_id) \
                FROM citations as c \
                    JOIN article_citations_context as acc ON c.id = acc.citation_id \
                    JOIN article as a ON acc.article_id = a.id \
                WHERE (a.nlp_css = 1 OR a.nlp_css_prediction = 1) AND (c.interdisciplinary = 1) AND (acc.paragraph_context IS NOT NULL);\
            "
    cur.execute(select_it_in_cited)
    cited_it_in_count = cur.fetchone()[0]
    print(f"Number of interdisciplinary citations: {interdisciplinary_count} ({cited_it_in_count} in-text)")



    # double check context and everything is there for 2024 and 2025 predictions
    select_context_not_null = "SELECT COUNT(*) FROM article as a \
          RIGHT JOIN article_citations as ac ON a.id = ac.article_id \
            LEFT JOIN citations as c ON ac.citation_id = c.id\
                WHERE (a.nlp_css_prediction = 1) AND (c.field_of_study IS NOT NULL) AND (a.year >= 2024);"
    cur.execute(select_context_not_null)
    print(f"\n2024-2025 article_citations: {cur.fetchone()}")


    # get unlabelled dataset
    select_articles = "SELECT COUNT(*) FROM article as a WHERE (a.nlp_css IS NULL)"
    cur.execute(select_articles)
    print(f"Number of unlabelled articles in dataset: {cur.fetchone()[0]}")

    select_positives = "SELECT COUNT(*) FROM article as a WHERE (a.nlp_css IS NULL) AND (a.nlp_css_prediction = 1)"
    cur.execute(select_positives)
    print(f"Number of positives in unlabelled set: {cur.fetchone()[0]}")


    # print some examples from the 
    select_example = "SELECT article_id, citation_id FROM article_citations_context where citation_text = 'Ng et al., 2024b';"
    cur.execute(select_example)
    print(cur.fetchone())

    select_example = "SELECT * FROM article_citations_context where article_id = 16744 AND citation_id = 38006;"
    cur.execute(select_example)
    for c in cur.fetchall():
        print(c)

    
    
    con.close()



if __name__ == "__main__":
    main()