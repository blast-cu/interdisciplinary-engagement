import os
import sqlite3
import pickle
import pandas as pd
import re

import tqdm

from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import xmls_to_dict
from data_utils.db_creation.utils import get_local_article_path, clean_sections

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def get_count(citation_id, article_id, cur):
    count_query = f"SELECT count FROM article_citations WHERE citation_id = {citation_id} AND article_id = {article_id}"
    cur.execute(count_query)
    count = cur.fetchone()[0]
    return count


def main():

    print("\n----------------------------------------------------")
    LOGGER.info("STEP 9: Adding section information to article_citations table...")

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    try:
        add_count_query = "ALTER TABLE article_citations ADD COLUMN count INT"
        cur.execute(add_count_query)
    except sqlite3.OperationalError:
        LOGGER.info("count column already exists")
    set_count_query = "UPDATE article_citations SET count = 0"
    cur.execute(set_count_query)

    # get all rows with no section
    get_no_section_query = "SELECT * FROM article_citations WHERE section IS NULL"
    cur.execute(get_no_section_query)
    no_section = cur.fetchall()
    before = len(no_section)
    LOGGER.info(f"{before} article_citations rows without section info...")

    try:
        add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN citation_text TEXT"
        cur.execute(add_ss_col_query)
    except sqlite3.OperationalError:
        LOGGER.info("citation_text column already exists")
    
    try:
        add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN context TEXT"
        cur.execute(add_ss_col_query)
    except sqlite3.OperationalError:
        LOGGER.info("context column already exists")

    try:
        add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN paragraph_context TEXT"
        cur.execute(add_ss_col_query)
    except sqlite3.OperationalError:
        LOGGER.info("paragraph_context column already exists")


    # get articles for which none of the section info in article_citations table is filled
    # get_articles_query = ""
    # cur.execute(get_articles_query)
    # articles_ids = cur.fetchall()

    # get id, pdf_id, venue, year for each article
    article_info_query = f"SELECT id, pdf_id, venue, year, nlp_css FROM article WHERE id IN (SELECT DISTINCT article_id FROM article_citations WHERE section IS NULL)"
    cur.execute(article_info_query)
    articles_info = cur.fetchall()


    for paper_id, pdf_id, venue, year, nlp_css in tqdm.tqdm(articles_info, desc="Adding section information for citations in articles"):

        data_path = get_local_article_path(venue, year, pdf_id, "dict", LOGGER, nlp_css)
        if data_path is None: 
            LOGGER.debug(f"Data path not found for article: {venue}, {year}, {pdf_id}")
            continue
    
        with open(data_path, 'rb') as f:
            p = pickle.load(f)

        sections = clean_sections(p['sections'], LOGGER)

        for section_title, sec in sections.items():
            sec_refs = sec['refs']
            sec_contexts = sec['contexts']
            sec_p_contexts = sec['paragraph_contexts']

            if len(sec_refs) != len(sec_contexts) or len(sec_refs) != len(sec_p_contexts):
                LOGGER.error(f"Refs and contexts not the same length: {len(sec_refs)}, {len(sec_contexts)}")
                exit()

            # get citations in paper
            citations_query = f"SELECT citations.year, citations.authors, citations.id, citations.title FROM citations \
                INNER JOIN article_citations ON citations.id = article_citations.citation_id WHERE article_citations.article_id = {paper_id}"

            
            cur.execute(citations_query)
            db_citations = cur.fetchall()

            if len(db_citations) == 0:
                LOGGER.debug(f'No citations found for paper: {venue}, {year}, {pdf_id}')
                continue

            db_citations = [(c[0], c[1].lower().strip(), c[2], c[3]) for c in db_citations]

            paper_citations = sec_refs
            for idx, citation in enumerate(paper_citations):

                try:
                    year = re.findall("[12][0-9]{3}", citation)[0]
                except IndexError as e:
                    # LOGGER.debug('Year not found in citation: ', citation, '\n')
                    continue

                author = citation.split(year)[0]

                if 'et al' in author:
                    author = author.split("et al")[0]
                elif 'and' in author:
                    author = author.split("and")[0]
                elif '&' in author:
                    author = author.split("&")[0]


                year = int(year)
                author = author.lower().strip()

                match = False
                for c in db_citations:
                    
                    db_year = c[0]
                    db_author = c[1].lower().strip()
                    citation_id = c[2]
                    if db_year == year:
                        if author in db_author:
                            match = True
                            section_title = section_title.replace("'", "''")
                            add_section_query = f"UPDATE article_citations SET section = '{section_title}' WHERE article_id = {paper_id} AND citation_id = {citation_id}"
                            cur.execute(add_section_query)
                            curr_ref = sec_refs[idx].replace("'", "''")
                            add_citation_text_query = f"UPDATE article_citations SET citation_text = '{curr_ref}' WHERE article_id = {paper_id} AND citation_id = {citation_id}"
                            cur.execute(add_citation_text_query)

                            curr_context = sec_contexts[idx].replace("'", "''")
                            add_context_query = f"UPDATE article_citations SET context = '{curr_context}' WHERE article_id = {paper_id} AND citation_id = {citation_id}"
                            cur.execute(add_context_query)

                            cur_p_context = sec_p_contexts[idx]
                            cur_p_context = cur_p_context.replace("'", "''")
                            add_p_context_query = f"UPDATE article_citations SET paragraph_context = '{cur_p_context}' WHERE article_id = {paper_id} AND citation_id = {citation_id}"
                            cur.execute(add_p_context_query)


                            count = get_count(citation_id, paper_id, cur)
                            count += 1
                            update_count_query = f"UPDATE article_citations SET count = {count} WHERE article_id = {paper_id} AND citation_id = {citation_id}"
                            cur.execute(update_count_query)
                            break

                        
    get_no_section_query = "SELECT * FROM article_citations WHERE section IS NULL"
    cur.execute(get_no_section_query)
    no_section = cur.fetchall()
    after = len(no_section)

    LOGGER.info(f"  added section information to {before - after} article_citations")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()