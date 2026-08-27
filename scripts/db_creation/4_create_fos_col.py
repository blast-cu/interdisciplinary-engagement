import os
import sqlite3
import pickle
import pandas as pd
import requests
import json
import re
import doi
import time
from tqdm import tqdm

from src.db_creation import clean_string
from src.utils import get_logger, load_env


FIELD_OF_STUDY = set()
SOURCE = 'semantic_scholar'

def handle_duplicate_citation(cur, title_clean, year, citation_id):

    # delete the citation with citation_id
    delete_query = f"DELETE FROM citations WHERE id = {citation_id}"
    cur.execute(delete_query)

    # update any article_citations that referenced citation_id to reference the existing citation
    get_existing_query = f"SELECT id FROM citations WHERE clean_title = '{title_clean}' AND year = {year}"
    cur.execute(get_existing_query)
    existing_id = cur.fetchone()[0]

    update_article_citations = f"UPDATE article_citations SET citation_id = {existing_id} WHERE citation_id = {citation_id}"
    try:
        cur.execute(update_article_citations)
    except sqlite3.IntegrityError as e:
        # if this causes a duplicate entry, we can just delete the duplicate article_citation
        if "UNIQUE constraint failed" in str(e):
            delete_duplicate_article_citation = f"DELETE FROM article_citations WHERE citation_id = {citation_id}"
            cur.execute(delete_duplicate_article_citation)
        else:
            raise e

def query_semantic_scholar(queries, headers, cur, con, logger):
    local_ids = [q[0] for q in queries]
    external_ids = [q[1] for q in queries]
    request = requests.post(  # https://www.semanticscholar.org/product/api#api-key-form
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params={'fields': ['fieldsOfStudy,year,authors,title,journal']},
        json={"ids": external_ids},
        headers=headers
    )
    if request.status_code == 200:
        results = request.json()
        if len(results) != len(queries):
            logger.error("Lengths don't match")
            logger.error(results)
            logger.error(queries)
            exit()
        for i, result in enumerate(results):
            if result is not None:

                update_clean_id = f"UPDATE citations SET clean_id = '{external_ids[i]}' WHERE id = {local_ids[i]}"
                cur.execute(update_clean_id)

                if result['year'] is not None:
                    year = result['year']
                    year = int(year)
                    update_query = f"UPDATE citations SET year = {year} WHERE id = {local_ids[i]}"
                    try:
                        cur.execute(update_query)
                    except Exception as e:
                        if "UNIQUE constraint failed: citations.clean_title, citations.year" in str(e):
                            get_title_query = f"SELECT clean_title FROM citations WHERE id = {local_ids[i]}"
                            cur.execute(get_title_query)
                            title_clean = cur.fetchone()[0]
                            handle_duplicate_citation(cur, title_clean, year, local_ids[i])
                        else:
                            raise e

                if result['authors'] is not None:
                    authors = [author['name'] for author in result['authors']]
                    authors = '; '.join(authors)
                    authors = clean_string(authors)
                    update_query = f"UPDATE citations SET authors = ? WHERE id = ?"
                    cur.execute(update_query, (authors, local_ids[i],))


                if result['title'] is not None:
                    title = clean_string(result['title'])
                    update_query = f"UPDATE citations SET title = ? WHERE id = ?"
                    cur.execute(update_query, (title, local_ids[i],))

                if result['journal'] is not None:
                    if 'name' in result['journal'].keys() and result['journal']['name'] is not None:
                        journal = result['journal']['name']
                        journal = clean_string(journal)
                        update_query = f"UPDATE citations SET journal = ? WHERE id = ?"
                        cur.execute(update_query, (journal, local_ids[i]))

                if result['fieldsOfStudy'] is not None:
                    field_of_study = result['fieldsOfStudy']
                    for field in field_of_study:
                        FIELD_OF_STUDY.add(field)
                    if len(field_of_study) > 1:
                        field_of_study = [str(r) for r in field_of_study]
                        field_of_study = '; '.join(field_of_study)
                    else:
                        field_of_study = str(field_of_study[0])

                    field_of_study = clean_string(field_of_study)

                    update_query = f"UPDATE citations SET field_of_study = ? WHERE id = ?"
                    cur.execute(update_query, (field_of_study, local_ids[i],))
                    update_query = f"UPDATE citations SET fos_source = ? WHERE id = ?"
                    cur.execute(update_query, (SOURCE, local_ids[i],))
                else:
                    logger.debug(f"No field of study for {external_ids[i]}")
                
            else:
                logger.debug(f"None result for {external_ids[i]}")
                # get_citation_query = f"SELECT * FROM citations WHERE id = {local_ids[i]}"
                # cur.execute(get_citation_query)
                # citation = cur.fetchone()
                # print(citation)

    else:  # If the request was unsuccessful, print the error code and message
        logger.error(external_ids)
        logger.error(request.status_code)
        logger.error(request.text)
        exit()


def main():

    LOGGER = get_logger("create_fos_col")
    env_vars = load_env()

    print("\n----------------------------------------------------")
    LOGGER.info("STEP 4: Adding field of study to citation table...")

    # Connect to the SQLite database
    con = sqlite3.connect(env_vars["DB_PATH"])
    cur = con.cursor()

    # get rows to iterate over
    get_nones_query = "SELECT id, id_type, external_id, clean_id FROM citations WHERE field_of_study IS NULL"
    cur.execute(get_nones_query)
    none_refs = cur.fetchall()
    before_count = len(none_refs)  # reccord null values before update
    LOGGER.info(f"Found {before_count} citations to add fos")

    # Directly define the API key (Reminder: Securely handle API keys in production environments)
    api_key = 'jqcybDbL211KkpTjrfqQi21tketCLJ8x5U0bm9hb'  # Replace with the actual API key

    # Define headers with API key
    headers = {'x-api-key': api_key}
    queries = []

    for id, id_type, external_id, clean_id in tqdm(none_refs, desc="Processing citations"):
        
        if clean_id != None:
            queries.append((id, clean_id))

        elif 'arxiv' in id_type.lower() or 'arxiv' in external_id.lower() or 'abs' in external_id:
            id_str = external_id
            external_id = re.findall("[0-9]{4}.[0-9]{4,5}", external_id)
            if external_id != []:
                external_id = external_id[0]
                external_id = f'ARXIV:{external_id}'
                queries.append((id, external_id))
        elif external_id is not None:
            external_id = external_id.replace('//', '/')
            external_id = f'DOI:{external_id}'
            external_id = doi.find_doi_in_text(external_id)

            if external_id is not None:
                external_id = doi.get_clean_doi(external_id)
                if external_id is not None:
                    queries.append((id, external_id))

        if len(queries) != 0 and len(queries) % 500 == 0:
            query_semantic_scholar(queries, headers, cur, con, LOGGER)
            queries = []
            
            # wait a bit to avoid rate limiting
            LOGGER.debug("Waiting to avoid rate limiting...")
            time.sleep(5)
            
    if len(queries) != 0:
        query_semantic_scholar(queries, headers, cur, con, LOGGER)

    get_after_nones_query = "SELECT * FROM citations WHERE field_of_study IS NULL"
    cur.execute(get_after_nones_query)
    after = cur.fetchall()
    after_count = len(after)
    LOGGER.info(f" added {before_count - after_count} field of study values")


    get_valid_id_no_fos_query = "SELECT * FROM citations WHERE field_of_study IS NULL AND clean_id IS NOT NULL"
    cur.execute(get_valid_id_no_fos_query)
    valid_id_no_fos = cur.fetchall()
    valid_id_no_fos_count = len(valid_id_no_fos)
    LOGGER.debug(f"Valid ID No FOS: {valid_id_no_fos_count}")


    con.commit()
    con.close()

if __name__ == '__main__':
    main()