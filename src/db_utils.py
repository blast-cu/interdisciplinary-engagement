import re
import sqlite3
from tqdm import tqdm
import logging

"""
Helper functions for working with the database.
"""

def create_connection(db_path: str) -> sqlite3.Connection:
    """
    returns connection to sqlite3 database
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    return conn, cur

def get_table_names(db_path: str) -> list:
    _, cur = create_connection(db_path)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [table_name[0] for table_name in cur.fetchall()]
    return table_names


def get_column_names(db_path: str, table_name: str) -> list:
    _, cur = create_connection(db_path)
    cur.execute(f"SELECT * FROM {table_name};")
    column_names = [description[0] for description in cur.description]
    return column_names

def clean_title(title: str) -> str:
    """
    standard cleaning function for all article titles
    """
    title = title.lower().strip()
    title = "_".join(title.split())
    title = re.sub('[^a-z_]+', '', title)
    return title

def add_column_ignore_duplicate(db_path: str, table_name: str):
    """
    add column to table, ignore if duplicate
    """
    conn, cur = create_connection(db_path)
    try:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN clean_title TEXT")
    
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            pass
        else:
            raise e
    conn.commit()


def add_clean_title_column(db_path: str, table_name: str, logger: logging.Logger) -> None:
    """
    use the standard cleaning function to create a 'clean_title' column based on 'title' column
    """
    conn, cur = create_connection(db_path)
    add_column_ignore_duplicate(db_path, table_name)

    id_query = f"SELECT id, title FROM {table_name}"
    cur.execute(id_query)
    rows = cur.fetchall()

    for row in tqdm(rows):
        id = row[0]
        title = row[1]
        clean = clean_title(title)
        update_query = f"UPDATE {table_name} SET clean_title = '{clean}' WHERE id = {id}"
        cur.execute(update_query)
    
    conn.commit()


def title_in_db(db_path: str, title: str, venue: str, year: int) -> bool:
    """
    Returns true if a title is already in the database for a given venue and year.
    """
    conn, cur = create_connection(db_path)
    in_db = False
    title_clean = clean_title(title)
    get_query = f"SELECT COUNT(*) FROM article WHERE clean_title = '{title_clean}' AND venue = '{venue}' AND year = {year}"
    cur.execute(get_query)
    count = cur.fetchone()[0]
    if count > 0:
        in_db = True

    conn.close()
    return in_db
