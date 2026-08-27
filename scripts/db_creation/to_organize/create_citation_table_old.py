import os
import sqlite3
import pickle
import pandas as pd


from data_utils.utils import DATA_DIR, DB_FILENAME
from data_utils.parse_pdf.utils import clean_title, xmls_to_dict
FIELD_OF_STUDY = set()

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
        title_empty += 1
        if external_id != '' and authors != "":
            first_author = authors.split(';')[0]
            first_author = authors.strip()
            title = f"untitled_{first_author}_{journal}"
            title_clean = clean_title(title)
            print(f"Title empty: {title}")
        else:
            return
    
    # insert_query = \
    #     f"INSERT INTO citations VALUES('{title_clean}', {id},'{title}','{id_type}', '{external_id}','{authors}', {year}, '{journal}', '{meeting}')"
    insert_query = f"""
        INSERT INTO citations (clean_title, id, title, id_type, external_id, authors, year, journal, meeting)
        VALUES('{title_clean}', {id},'{title}','{id_type}', '{external_id}','{authors}', {year}, '{journal}', '{meeting}')
    """

    try:
        cur.execute(insert_query)
        count += 1
    except sqlite3.IntegrityError as e:
        print(e)
        get_duplicate_query = f"SELECT * FROM citations WHERE clean_title='{title_clean}'"
        cur.execute(get_duplicate_query)
        old_values = cur.fetchone()
        
        maybe_values = [title_clean, id, title, id_type, external_id, authors, year, journal, meeting]
        new_values = []
        for i, old in enumerate(old_values):
            if old == "":
                new_values.append(maybe_values[i])
            else:
                new_values.append(old)
        
        for i, v in enumerate(new_values):
            if type(v) == str:
                new_values[i] = v.replace("'", "''")
            
        
        remove_query = f"DELETE FROM citations WHERE clean_title='{title_clean}'"
        cur.execute(remove_query)
    
        insert_query = \
            f"INSERT INTO citations VALUES('{new_values[0]}', {new_values[1]},'{new_values[2]}','{new_values[3]}', '{new_values[4]}','{new_values[5]}', {new_values[6]}, '{new_values[7]}', '{new_values[8]}')"
        cur.execute(insert_query)
        
        id = old_values[1]
    
    insert_query = \
        f"INSERT INTO article_citations (article_id, citation_id) VALUES({article_id}, {id})"

    try:
        cur.execute(insert_query)
    except sqlite3.IntegrityError as e:
        pass

    return count

def main():
    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    create_table_query = \
        "CREATE TABLE citations(\
            clean_title TEXT, \
            id INTEGER NOT NULL,\
            title TEXT NOT NULL,\
            id_type TEXT NOT NULL,\
            external_id TEXT NOT NULL,\
            authors TEXT NOT NULL,\
            year INT,\
            journal TEXT,\
            meeting TEXT,\
            PRIMARY KEY(clean_title, year)\
        )"
    try:
        cur.execute(create_table_query)
    except:  # table already exists
        pass

    # delete_table_query = "DROP TABLE IF EXISTS article_citations"
    # cur.execute(delete_table_query)

    create_table_query = \
        "CREATE TABLE article_citations(\
            article_id INT NOT NULL,\
            citation_id INT NOT NULL,\
            section TEXT,\
            FOREIGN KEY(article_id) REFERENCES article(id),\
            FOREIGN KEY(citation_id) REFERENCES citations(id),\
            PRIMARY KEY(article_id, citation_id)\
        )"
    try:
        cur.execute(create_table_query)
    except:  # table already exists
        pass

    count = 1
    title_empty = 0

    df = pd.read_csv(f"{DATA_DIR}venue_info.csv")
    for ind in df.index:

        venue_year = df['year'][ind]
        venue = df['venue'][ind]

        

        print(f"Venue: {venue}, Year: {venue_year}")
        no_id_count = 0
        for cat in ['predicted_pos', 'pos']:
            xml_dir = f"{DATA_DIR}venues/{venue}/{venue_year}/{cat}/xml/"
            dict_dir = f"{DATA_DIR}venues/{venue}/{venue_year}/{cat}/dict/"

            if not os.path.exists(xml_dir):
                if cat == 'pos':
                    print(f"Positive directory doesn't exist")
                    continue
                else:
                    if venue == 'nlp_css':
                        continue
                    else:
                        print(f"Directory {dict_dir} does not exist yet")
                        exit()

            # xmls_to_dict(xml_dir, dict_dir)

            pickles = os.listdir(dict_dir)
            for p in pickles:
                if p.endswith('.pkl'):
                    curr = pickle.load(open(f"{dict_dir}{p}", "rb"))
                    # print("Document: ", p)

                    pdf_id = p.split('.')[0]
                    article_id_query = f"SELECT id FROM article WHERE pdf_id = {pdf_id} AND venue = '{venue}' AND year = {venue_year}"
                    cur.execute(article_id_query)
        
                    try:
                        article_id = cur.fetchone()[0]
                    except:
                        print(pdf_id)
                        print(venue_year)
                        print(venue)
                        query = f"SELECT pdf_id FROM article WHERE venue = '{venue}' AND year = {venue_year}"
                        cur.execute(query)
                        pdf_ids = cur.fetchall()
                        print(f"pdf_ids: {pdf_ids} found in the database")
                        exit()

                    for r in curr['references']:
                        count = add_citations_to_db(cur, article_id, r, venue_year, count)

                        
    cur.execute("ALTER TABLE citations RENAME TO old_table;")

    create_table_query = \
        "CREATE TABLE citations(\
            id INTEGER PRIMARY KEY,\
            title TEXT NOT NULL,\
            id_type TEXT NOT NULL,\
            external_id TEXT NOT NULL,\
            authors TEXT NOT NULL,\
            year INT NOT NULL,\
            journal TEXT,\
            meeting TEXT,\
            field_of_study TEXT,\
            fos_source TEXT,\
            clean_id TEXT\
        )"
    cur.execute(create_table_query)
    move_data_query = \
        "INSERT INTO citations(id, title, id_type, external_id, authors, year, journal, meeting) \
            SELECT id, title, id_type, external_id, authors, year, journal, meeting FROM old_table;"
    cur.execute(move_data_query)

    delete_table_query = "DROP TABLE IF EXISTS old_table"
    cur.execute(delete_table_query)

    cur.execute("SELECT * FROM citations")
    # print(f"Length of citation table: {len(cur.fetchall())}")
    print(cur.description)
    print('length: ', len(cur.fetchall()))
    print(f"Title empty: {title_empty}")
    
    con.commit()
    con.close()
                

if __name__ == "__main__":
    main()