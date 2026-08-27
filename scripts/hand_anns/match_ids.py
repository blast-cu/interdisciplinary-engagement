import pandas as pd
import logging
from tqdm import tqdm
import numpy as np

from src.utils import load_env, get_logger
from src.db_utils import create_connection
from src.db_creation import clean_string

def get_jaccard_index(a, b):
    """
    Calculate the Jaccard index between two sets.
    """
    if not a or a is np.nan or not b or b is np.nan:
        return np.nan
    
    # if a and b are not lists, convert them to sets
    if isinstance(a, str):
        a = a.lower().strip().split(" ")

    if isinstance(b, str):
        b = b.lower().strip().split(" ")

    a = set(a)
    b = set(b)

    intersection = len(a.intersection(b))
    union = len(a.union(b))

    if union == 0:
        return 0
    else:
        jaccard_index = intersection / union
        return jaccard_index

def articles_equal(db_res, df_row):
    art_id, clean_title, venue, year = db_res
    assert art_id == df_row["id"].item(), "ids not equal"
    assert clean_title == df_row["clean_title"].item(), "clean titles not equal"
    assert venue == df_row["venue"].item(), "venues not equal"
    assert year == df_row["year"].item(), "years not equal"

def article_parsed(db_res, cur):
    art_id, clean_title, venue, year = db_res
    select_citations_query = f"SELECT * FROM article_citations_context WHERE article_id = '{art_id}';"
    cur.execute(select_citations_query)
    res = cur.fetchall()
    assert len(res) > 0, f"No entries for {art_id}: {clean_title}, {venue}, {year}"


def assert_all_articles_equal(art_df: pd.DataFrame, art_db_results: list, cur, logger: logging.Logger):

    for (art_id, clean_title, venue, year) in art_db_results:
        df_row = art_df[art_df["id"] == art_id]
        articles_equal((art_id, clean_title, venue, year), df_row)
        article_parsed((art_id, clean_title, venue, year), cur)

    logger.info("All articles matched and parsed")

def get_max_sim_context(cur_context, results):
    max_sim, cit_id = -1, -1
    for cur_res in results:
        cur_sim = get_jaccard_index(cur_res[3], cur_context)
        if cur_sim > max_sim:
            max_sim = cur_sim
            cit_id = (cur_res[0], cur_res[1], cur_res[2])
    
    return max_sim, cit_id


def get_citation_match(df_row, cur, logger):
    """
    need article_id, citation_id, citation_idx from article_citations_context table
    """
    art_id = df_row["article_id"]
    cit_text = clean_string(df_row["citation_text"])
    cit_context = clean_string(df_row["citation_context"])

    # print(cit_text)

    # need to get the index from the database
    filtered = True
    select_citations_query = "SELECT article_id, citation_id, citation_idx, paragraph_context FROM article_citations_context WHERE article_id = ? AND citation_text = ?;"
    cur.execute(select_citations_query, (art_id, cit_text,))
    res = cur.fetchall()
    
    # assert len(res) > 0, f"No entries for article {art_id}: '{cit_text}'"
    if len(res) == 0:
        filtered = False
        select_all_citations_query = "SELECT article_id, citation_id, citation_idx, paragraph_context FROM article_citations_context WHERE citation_text = ?"
        cur.execute(select_all_citations_query, (cit_text,))
        res = cur.fetchall()


    max_sim, cit_context_id = get_max_sim_context(cit_context, res)
    
    # assert max_sim > 0.9, f"Similarity {max_sim} too low to validate match"
    if max_sim < 0.9:
        
        if filtered: # remove filter, try again
            select_all_citations_query = f"SELECT article_id, citation_id, citation_idx, paragraph_context FROM article_citations_context"
            cur.execute(select_all_citations_query)
            res = cur.fetchall()

            max_sim, cit_context_id = get_max_sim_context(cit_context, res)
            if max_sim < 0.7:
                return max_sim
        
        else: # we looked everywhere
            return max_sim
            
    return cit_context_id

    


def main():
    art_path = "data/articles_old_ids.csv"
    cit_path = "data/citations_old_ids.csv"

    logger = get_logger("match_ann_ids")
    env_vars = load_env()

    # load annotations and db
    art_df = pd.read_csv(art_path)
    cit_df = pd.read_csv(cit_path)
    con, cur = create_connection(env_vars["DB_PATH"])

    # first make sure all of the articles are in the db
    art_ids = art_df["id"].to_list()
    id_placeholders = ', '.join(['?'] * len(art_ids))
    select_arts_from_ids_query = f"SELECT id, clean_title, venue, year FROM article WHERE id IN ({id_placeholders});"
    cur.execute(select_arts_from_ids_query, art_ids)
    selected_articles = cur.fetchall()
    assert len(selected_articles) == len(art_ids), "Did not fetch all articles from the database"

    # check that the article content is the same
    assert_all_articles_equal(art_df, selected_articles, cur, logger)

    # that passed, so we can just use the article ids in the citations df
    # we know the ids have changed, so we need to match based on context info
    errors = []
    citations = []
    for idx, df_row in tqdm(cit_df.iterrows(), total=len(cit_df), desc="Matching citations"):
        cit_context_id = get_citation_match(df_row, cur, logger)
        if type(cit_context_id) is float or type(cit_context_id) is int:
            errors.append({
                "article_title": "",
                "max_sim": cit_context_id,
                "citation_text": df_row["citation_text"],
                "citation_context": df_row["citation_context"]
            })
        else: 
            citations.append({ # citation_id,article_id,citation_text,section,category,ann1,ann2,ann3,citation_context
                "article_id": cit_context_id[0],
                "citation_id": cit_context_id[1],
                "citation_idx": cit_context_id[2],
                "citation_text": df_row["citation_text"],
                "section": df_row["section"],
                "category": df_row["category"],
                "ann1": df_row["ann1"],
                "ann2": df_row["ann2"],
                "ann3": df_row["ann3"],
                "citation_context": df_row["citation_context"]

            })


    logger.error(f"Could not match {len(errors)} citations")

    # for e_row in errors:
    #     article_id = e_row["article_id"]
    #     select_art_info = f"SELECT venue, year, clean_title FROM article WHERE id = '{article_id}'"
    #     print(cur.execute(select_art_info).fetchone())
    #     print(e_row)
    #     print()
    pd.DataFrame.from_dict(citations).to_csv("citations.csv")
    pd.DataFrame.from_dict(errors).to_csv("error_citations.csv")
    
    
    con.close()


if __name__ == "__main__":
    main()