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

    matplotlib.style.use('seaborn-v0_8-darkgrid')
    rcParams.update({'figure.autolayout': True})

    # get all rows with no section
    to_predict = {}
    # get_no_section_query = "SELECT ac.arg_pred, \
    #     citations.interdisciplinary FROM article_citations AS ac  \
    #      JOIN citations ON ac.citation_id = citations.id \
    #         WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and social_science = 1;"

    # df = pd.read_sql_query(get_no_section_query, con)
    # df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    # df = df.groupby('arg_pred').size()
    # df.plot.pie(y='count', autopct="%.1f%%", explode=(0.01, 0.01, 0.01), figsize=(5, 5), colors=['#235789', '#F2B134', '#D5573B'])
    # plt.title('Percentage of Argument Types Supported\n by Social Science Citations')
    # plt.savefig(f"{ANALYSIS_DIR}/ss_pie_arg_types.png", dpi=300)
    # plt.show()

    get_no_section_query = "SELECT ac.arg_pred, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and interdisciplinary = 1;"

    df = pd.read_sql_query(get_no_section_query, con)
    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.groupby('arg_pred').size()
    df.plot.pie(y='count', fontsize=17, autopct="%.1f%%", explode=(0.01, 0.01, 0.01), figsize=(5, 5), colors=['#235789', '#F2B134', '#D5573B'])
    plt.title('Out-of-Discipline Citations', fontsize=14)
    plt.savefig(f"{ANALYSIS_DIR}/inter_pie_arg_types.png", dpi=300)
    plt.show()

    get_no_section_query = "SELECT ac.arg_pred, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and interdisciplinary != 1;"
    plt.figure(figsize=(10, 5))
    df = pd.read_sql_query(get_no_section_query, con)
    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.groupby('arg_pred').size()
    df.plot.pie(y='count', fontsize=17, autopct="%.1f%%", explode=(0.01, 0.01, 0.01), figsize=(5, 5), colors=['#235789', '#F2B134', '#D5573B'])
    plt.title('In-Discipline Citations', fontsize=14)
    plt.savefig(f"{ANALYSIS_DIR}/in_disc_pie_arg_types.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()