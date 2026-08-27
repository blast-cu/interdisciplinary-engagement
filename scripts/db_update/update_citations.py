from tqdm import tqdm
import pickle
import re
import sqlite3

from src.utils import load_env, get_logger
from src.db_creation import add_citations_to_db, clean_sections, clean_string, get_local_article_path
from src.db_utils import create_connection, get_table_names, get_column_names

"""
This script updates the existing database by removing the article_citations table and replacing it 
with new parsing schema which accounts for citation frequency
"""

def get_ref_context_from_article(cur, env_vars, venue, year, paper_id, logger):
    """
    process the references from a single article
    """
    # open dictionary for parsed pdf of article
    data_path = get_local_article_path(env_vars["VENUE_DATA_PATH"], venue, year, paper_id, logger)
    with open(data_path, 'rb') as f:
        p = pickle.load(f)

    # get citations from the database that are in the current paper
    citations_query = f"SELECT citations.year, citations.f_author_ln, citations.authors, citations.id, citations.title FROM citations \
        INNER JOIN article_citations ON citations.id = article_citations.citation_id WHERE article_citations.article_id = {paper_id}"

    cur.execute(citations_query)
    db_article_citations = cur.fetchall()

    # attempt to add citations to article_citations if none found
    if len(db_article_citations) == 0:
        logger.warning(f'No citations found for paper: {venue}, {year}, {paper_id}, trying to add them')
        citations_added = add_citations_to_db(cur, data_path, paper_id, year, logger)
        if citations_added > 0:
            get_ref_context_from_article(cur, env_vars, venue, year, paper_id, logger)
        else:
            logger.warning('Not successful')
            return cur, 0

    db_citations = {}
    for c in db_article_citations:
        # try to fetch nav_id 
        select_nav_id_query = "SELECT nav_id FROM article_citations WHERE article_id = ? and citation_id = ?;"
        cur.execute(select_nav_id_query, (paper_id, c[3]))
        nav_id = cur.fetchone()[0]

        # add everything to a dict
        c_id = c[3]
        db_citations[c_id] = {
            "year": c[0], "f_author_ln": c[1].lower(), "authors": c[2].lower().strip(), 
            "title": c[4], "nav_id": nav_id
        }
    nav_id_map = {v["nav_id"]: k for k, v in db_citations.items()}


    # get dictionary of sections where key is cannonical section and value is dict of refs, contexts, and paragraph contexts
    sections = clean_sections(p['sections'], logger)
    successful_count = 0
    for section_title, sec in sections.items():
        sec_refs = sec['refs']
        sec_contexts = sec['contexts']
        sec_p_contexts = sec['paragraph_contexts']
        sec_nav_ids = sec['nav_ids']

        # the number of references need to match the number of contexts
        if len(sec_refs) != len(sec_contexts) or len(sec_refs) != len(sec_p_contexts) or len(sec_refs) != len(sec_nav_ids):
            logger.error(f"Refs and contexts not the same length: {len(sec_refs)}, {len(sec_contexts)}, {len(sec_nav_ids)}")
            raise ValueError

        # map each in-text citation we found to the corresponding citation 
        for idx, ref in enumerate(sec_refs):
            # prep ref text to be added to table
            year, author, alpha_idx = clean_ref(ref, logger)
            author = clean_string(author)

            cur_ref_text = clean_string(ref)
            cur_context = clean_string(sec_contexts[idx])
            cur_p_context = clean_string(sec_p_contexts[idx])
            cur_nav_id = sec_nav_ids[idx]

            if cur_ref_text not in cur_p_context:
                print(f"\nref text: '{cur_ref_text}' not in context: '{cur_context}'")
                print(f"paragraph: {cur_p_context}")
                raise ValueError

            authors = []

            if cur_nav_id in nav_id_map: 
                c_match_id = nav_id_map[cur_nav_id]
                add_article_info_context_entry(
                    cur, paper_id, c_match_id, section_title, 
                    cur_ref_text, cur_context, cur_p_context
                )
                successful_count += 1
            
            elif alpha_idx: # not handling the alpha index at the end of the year in some citations
                continue

            else: # loop over the citations
                # handle multiple matches
                matches = []
                for citation_id, c in db_citations.items():
                    
                    first_author_ln = clean_string(c["f_author_ln"])
                    authors.append(first_author_ln)

                    if c["year"] == year:
                        if author == first_author_ln: # match!!
                            matches.append(citation_id)

                if len(matches) == 1: # just one match, straightforward
                    add_article_info_context_entry(
                        cur, paper_id, matches[0], section_title, 
                        cur_ref_text, cur_context, cur_p_context
                    )
                    successful_count += 1

    return cur, successful_count

def get_count(citation_id, article_id, cur):
    count_query = f"SELECT count FROM article_citations WHERE citation_id = {citation_id} AND article_id = {article_id};"
    try:
        cur.execute(count_query)
        count = cur.fetchone()[0]

    except Exception as e: 
        print(count_query)
        raise e

    return count

def add_article_info_context_entry(
        cur, article_id, citation_id, section_title, citation_text, context, paragraph_context
):

    # update count in article_citations table, get index for article_citations_context table
    citation_idx = get_count(citation_id, article_id, cur)
    count = citation_idx + 1
    update_count_query = f"UPDATE article_citations SET count = ? WHERE article_id = ? AND citation_id = ?"
    cur.execute(update_count_query, (count, article_id, citation_id, ))

    # add row in article_citations_context table
    insert_row_query = "INSERT INTO article_citations_context (article_id, citation_id, citation_idx, section, citation_text, context, paragraph_context) VALUES (?, ?, ?, ?, ?, ?, ?)"
    row_data = (article_id, citation_id, citation_idx, section_title, citation_text, context, paragraph_context)

    try:
        cur.execute(insert_row_query, row_data)
    except Exception as e:
        raise e

    return

def clean_ref(citation, logger):
    citation = re.sub('[();]', '', citation)
    alpha_idx = None
    try:
        year = re.findall("[12][0-9]{3}[a-z]", citation)
        if len(year) == 0:
            year = re.findall("[12][0-9]{3}", citation)
            year = year[0]
            alpha_idx = None
        else:
            year = year[0]
            alpha_idx = year[-1]
            year = year[:-1]
    except IndexError as e:
        logger.debug('Year not found in citation: ', citation, '\n')
        return None, None, None

    author = citation.split(year)[0]

    if 'et al' in author:
        author = author.split("et al")[0]
    elif 'and' in author:
        author = author.split("and")[0]
    elif '&' in author:
        author = author.split("&")[0]

    year = int(year)
    author = author.lower().replace(",", "").replace(".", "").strip()

    return (year, author, alpha_idx)


def main():

    logger = get_logger("parse_citations")
    env_vars = load_env()
    
    con, cur = create_connection(env_vars["DB_PATH"])

    # set frequencies to 0
    cur.execute("UPDATE article_citations SET count = 0;")
    con.commit()

    # drop article_citations_context if it exists
    cur.execute("DROP TABLE IF EXISTS article_citations_context")

    # create a fresh table
    create_acc_table_query = \
            "CREATE TABLE article_citations_context(\
                article_id INT NOT NULL,\
                citation_id INT NOT NULL,\
                citation_idx INT NOT NULL,\
                section TEXT,\
                citation_text TEXT,\
                context TEXT,\
                paragraph_context TEXT,\
                FOREIGN KEY(article_id) REFERENCES article_citations(article_id),\
                FOREIGN KEY(citation_id) REFERENCES article_citations(citations_id),\
                PRIMARY KEY(article_id, citation_id, citation_idx)\
            )"
    cur.execute(create_acc_table_query)
    con.commit()
    logger.info(f"Created new article_citation_context table at {env_vars['DB_PATH']}")


    logger.info("Adding section information to article_citations table...")

    # get all the articles that need to be parsed 
    # get id, pdf_id, venue, year for each article
    article_info_query = f"SELECT id, venue, year FROM article WHERE (nlp_css = 1 OR nlp_css_prediction = 1)"
    cur.execute(article_info_query)
    articles_info = cur.fetchall()
    logger.info(f"Found {len(articles_info)} articles to parse") # expect 2052

    articles_missing_citations = []
    for paper_id, venue, year in tqdm(articles_info, desc="Processing in-text citations in each article"):

        # skip papers with known parse errors
        if paper_id in [1690, 2498, 4230, 4458, 8729, 11568, 11899, 12715, 18029]:
            continue
        
        cur, succsessful_count = get_ref_context_from_article(
            cur, env_vars, venue, year, paper_id, logger
        )
        con.commit()

        if succsessful_count == 0: # article doesn't have citations
            print(paper_id)
            articles_missing_citations.append(paper_id)
            # 75, 2620, 6677,  -> authors didn't use \cite{}, can't identify in-text citations in article
            # 8369, 17887, 17869  -> parse errors (extra space in-text citations???)
            # 17965 -> one column format?
            # 
            # 16133, 13300, 13043, 12805
                            
    logger.warning(f"{len(articles_missing_citations)} articles don't have citations")

    # get the number of rows in the article_citations table 
    cur.execute("SELECT COUNT(*) FROM article_citations;")
    ac_count = cur.fetchone()[0]

    # get the number of rows in article_citations_context table
    cur.execute("SELECT COUNT(*) FROM article_citations_context;")
    logger.info(f"Now {cur.fetchone()[0]} rows in article_citations_context table ({ac_count} rows in article_citations)")

    con.commit()
    con.close()


if __name__ == "__main__":
    main()


