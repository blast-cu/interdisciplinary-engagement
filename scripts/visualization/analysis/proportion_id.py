import pickle
import pandas as pd
import re
import sqlite3
from data_utils.utils import DATA_DIR, DB_FILENAME
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import sqlite3
import os
import pandas as pd

from data_utils.utils import DB_FILENAME
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

con = sqlite3.connect(DB_FILENAME)
cur = con.cursor()

def get_year(row):
    cur.execute(f"SELECT year FROM article WHERE id = {row['article_id']};")
    return cur.fetchone()[0]

def get_conference(row):
    cur.execute(f"SELECT venue FROM article WHERE id = {row['article_id']};")
    return cur.fetchone()[0]

def main():

    # get all rows with no section
    to_predict = {}
    get_no_section_query = "SELECT ac.article_id, ac.citation_id, ac.section, citations.field_of_study, citations.social_science, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO';"
    df = pd.read_sql_query(get_no_section_query, con)
    
    df['year'] = df.apply(get_year, axis=1)
    df['not_interdisciplinary'] = df.apply(lambda x: 1 if x['interdisciplinary'] == 0 else 0, axis=1)
    df['count'] = 1
    df['venue'] = df.apply(get_conference, axis=1)

    # year_plot = df.groupby('article_id').agg({'not_interdisciplinary': 'sum', 'interdisciplinary': 'sum', 'year': 'max'}) 
    matplotlib.style.use('seaborn-v0_8-darkgrid')
    rcParams.update({'figure.autolayout': True})
    # year_plot = year_plot.groupby(['year']).mean().plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    # plt.xlabel('Year')
    # plt.ylabel('Average Number of Citations')
    # plt.title('Average Number of Interdisciplinary Citations \nin a Publication by Year Published')
    # plt.legend(["Not Interdisciplinary", "Interdisciplinary"])
    
    # plt.savefig(f"{ANALYSIS_DIR}/avg_interdisciplinary_citations_by_year.png", dpi=300)
    # plt.show()


    venue_plot = df.groupby('article_id').agg({'venue': 'max', 'not_interdisciplinary': 'sum', 'interdisciplinary': 'sum'})
    venue_plot = venue_plot.replace("nlp_css", "NLP+CSS")
    venue_plot = venue_plot.groupby(['venue']).mean().plot(kind='bar', stacked=True, color=['#235789', '#D5573B'])
    plt.xlabel('Venue')
    plt.ylabel('Average Number of Citations')
    plt.title('Average Number of Interdisciplinary Citations \nin a Publication by Conference')
    plt.legend(["Not Interdisciplinary", "Interdisciplinary"])
    
    plt.savefig(f"{ANALYSIS_DIR}/avg_interdisciplinary_citations_by_venue.png", dpi=300)
    plt.show()


    
    
if __name__ == "__main__":
    main()