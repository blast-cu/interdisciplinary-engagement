import sqlite3

from src.utils import load_env
from src.db_creation import clean_string, clean_db_string

"""
clean the text in the article and citation tables
"""

def main():
    env_vars = load_env()

    # Connect to your database
    conn = sqlite3.connect(env_vars["DB_PATH"])

    # 2. Register the Python function as an SQL function
    # Arguments: (SQL function name, number of arguments, Python callable)
    conn.create_function("py_clean", 1, clean_string)

    # 3. Apply the function to all relevant text columns
    # An UPDATE without a WHERE clause affects every row
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE citations 
        SET 
            title = py_clean(title),
            authors = py_clean(authors),
            journal = py_clean(journal),
            meeting = py_clean(meeting),
            field_of_study = py_clean(field_of_study),
            fos_source = py_clean(fos_source)
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()