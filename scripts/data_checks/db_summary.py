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

    print("\nTrack Label Summary (hand-annotated dataset): ")
    select_labels_query = "SELECT nlp_css, COUNT(id) FROM article GROUP BY nlp_css"
    cur.execute(select_labels_query)
    print(cur.fetchall())

    print("\nPredictions: ")
    select_predictions_query = "SELECT nlp_css_prediction, COUNT(article_id) FROM predictions GROUP BY nlp_css_prediction;"
    cur.execute(select_predictions_query)
    print(cur.fetchall())

    # get the entire positive set
    select_pred_pos_ids = "SELECT article_id FROM predictions WHERE nlp_css_prediction = 1;"
    cur.execute(select_pred_pos_ids)
    pred_pos_ids = [entry[0] for entry in cur.fetchall()]
    
    select_labeled_pos_ids = "SELECT id FROM article WHERE nlp_css = 1;"
    cur.execute(select_labeled_pos_ids)
    labeled_pos_ids = [entry[0] for entry in cur.fetchall()]

    nlp_css_ids = list(set(pred_pos_ids + labeled_pos_ids))
    print("\nSet of NLP+CSS articles")
    print(f"- predicted positive: {len(pred_pos_ids)}")
    print(f"- labeled positive: {len(labeled_pos_ids)}")
    print(f"- total: {len(nlp_css_ids)}")


    # select all articles, see what info they have
    id_placeholder = ', '.join(['?'] * len(nlp_css_ids))
    select_nlp_css_articles = f"SELECT * FROM article WHERE id IN ({id_placeholder});"
    cur.execute(select_nlp_css_articles, nlp_css_ids)
    print(f"\nArticle info fetched for {len(cur.fetchall())} articles")

    # check how many have entries in the article_citation_context table
    select_article_citations = "SELECT article_id, citation_id FROM article_citations_context AS acc;"
    cur.execute(select_article_citations)
    res = cur.fetchall()
    
    unique_articles = list(set([r[0] for r in res]))
    print(f"Fetched {len(res)} citation context entries from {len(unique_articles)} articles")


    # check that frequncies are right
    ac_freq_query = "SELECT article_id, citation_id, count FROM article_citations;"
    cur.execute(ac_freq_query)
    ac_freq = cur.fetchall()

    for a_id, c_id, count in ac_freq:
        
        # get number of entries in article_citations_context
        context_query = "SELECT citation_idx FROM article_citations_context WHERE article_id = ? AND citation_id = ?;"
        cur.execute(context_query, (a_id, c_id))
        citation_indices = [r[0] for r in cur.fetchall()]

        assert len(citation_indices) == count, "incorrect frequency value"

        if count != 0:
            max_idx = max(citation_indices)
            assert (max_idx + 1) == count, "incorrect frequency or citation idx"


    con.close()


if __name__ == "__main__":
    main()