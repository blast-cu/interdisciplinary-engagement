import argparse
import sqlite3
import os
import pickle
from tqdm import tqdm

from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.db_creation.utils import add_clean_title_column
from data_utils.parse_pdf.utils import clean_title

def add_citations_to_db(cur, article_id, reference, year, count):
    r = reference
    title = r['title'].replace("'", "''")
    title_clean = clean_title(title)
    id = count
    id_type = r['id_type'].replace("'", "''")
    external_id = r['id'].replace("'", "''")
    authors = r['authors'].replace("'", "''")
    journal = r['journal'].replace("'", "''")
    meeting = r['meeting'].replace("'", "''")

    year = r['year']
    if year == '' or year is None:
        year = -1
    elif '-' in year:
        year = year.split('-')[0]
    year = int(year)

    if title == '':
        # title_empty += 1
        # print("Title empty")
        if external_id != '' and authors != "":
            first_author = authors.split(';')[0]
            first_author = authors.strip()
            title = f"untitled_{first_author}_{journal}"
            title_clean = clean_title(title)
            # print(f"Title empty: {title}")
        else:
            return
    
    # insert_query = \
    #     f"INSERT INTO citations VALUES('{title_clean}', {id},'{title}','{id_type}', '{external_id}','{authors}', {year}, '{journal}', '{meeting}')"
    insert_query = f"""
        INSERT INTO citations (clean_title, title, id_type, external_id, authors, year, journal, meeting)
        VALUES('{title_clean}', '{title}','{id_type}', '{external_id}','{authors}', {year}, '{journal}', '{meeting}')
    """

    try:
        cur.execute(insert_query)
        id = cur.lastrowid

    except sqlite3.IntegrityError as e:
        print(e)
        get_duplicate_query = f"SELECT * FROM citations WHERE id = {id}"
        cur.execute(get_duplicate_query)
        old_values = cur.fetchone()

        print(old_values)
        exit()
        
    
    insert_query = \
        f"INSERT INTO article_citations (article_id, citation_id) VALUES({article_id}, {id})"

    # try:
    cur.execute(insert_query)
    # except sqlite3.IntegrityError as e:
    #     pass


def main(args):
    year = args.y
    # year = 2023

    # read in sighum papers
    dict_dir = f"{DATA_DIR}sighum/{year}/dict/"  # path to pickled dicts

    ### CHECK IF IN ARTICLES TABLE ALREADY
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # remove all sighum entries first
    delete_query = f"DELETE FROM article WHERE venue = 'sighum' and year = {year}"
    cur.execute(delete_query)

    # create clean title column if it does not exist
    add_clean_title_column(con, 'article')
    add_clean_title_column(con, 'citations')

    # get number of rows in citations table
    cur.execute("SELECT COUNT(*) FROM citations")
    citations_count = cur.fetchone()[0] + 1  # start from next id

    for file in tqdm(os.listdir(dict_dir), desc="Processing SIGHUM papers"):
        if file.split(".")[-1] != "pkl":
            continue
        pdf_id = int(file.split(".")[0])
        with open(dict_dir + file, "rb") as f:

            data = pickle.load(f)
            title = data["title"]
            title_clean = clean_title(title)

            # match article id with title
            # find title match
            clean = clean_title(title)
            match_query = f"SELECT id FROM article WHERE clean_title = '{clean}' AND year = {year}"
            cur.execute(match_query)
            article_id = cur.fetchone()
            if article_id is not None:  # match found
                
                article_id = article_id[0]
                # set venue to sighum
                update_venue_query = f"UPDATE article SET venue = 'sighum' WHERE id = {article_id}"
                # print(update_venue_query)
                cur.execute(update_venue_query)

                # set nlp_css value to 1 or 0
                cur.execute(f"UPDATE article SET nlp_css = 1 WHERE id = {article_id}")
                # print(f"Updated existing article id {id} with venue sighum and nlp_css = 1")

            else:
                # add to articles table with nlp_css = 1
                title_clean = clean_title(title)
                title = title.replace("'", "''")
                
                abstract = data["abstract"].strip()
                if abstract.endswith(". 1"):
                    abstract = abstract[:-2]
                abstract = abstract.replace("'", "''")

                insert_query = \
                    f"INSERT INTO article(clean_title, pdf_id, title, abstract, venue, year, nlp_css)\
                            VALUES('{title_clean}', {pdf_id}, '{title}', '{abstract}', 'sighum', {year}, 1)"
                cur.execute(insert_query)
                article_id = cur.lastrowid

            # now remove all entries from the citations table for this article id
            delete_query = f"DELETE FROM article_citations WHERE article_id = {article_id}"
            cur.execute(delete_query)

            # print(f"Adding citations for article id {article_id}, title: {title}")
            # add to citations table
            for r in data['references']:
                citations_count = add_citations_to_db(cur, article_id, r, year, citations_count)

            


    con.commit()
    con.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add SIGHUM papers to the database")
    parser.add_argument("--y", type=int, required=True, help="Year of the SIGHUM papers to add")
    args = parser.parse_args()

    main(args)