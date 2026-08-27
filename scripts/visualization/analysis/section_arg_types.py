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
    get_no_section_query = "SELECT ac.arg_pred, ac.section, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and social_science = 1;"
    

    df = pd.read_sql_query(get_no_section_query, con)
    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.replace({'section': {'introduction': 'Introduction', 'related work': 'Related Work', 'method': 'Method', 'experiments': 'Experiments', 'conclusion': 'Conclusion', 'appendix': 'Appendix'}})
    sections = ['Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion', 'Appendix']
    mapping = {s: i for i, s in enumerate(sections)}
    key = df['section'].map(mapping)

    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.groupby(['section','arg_pred']).size().unstack().reindex(sections)\
                                        .plot(kind='bar',color=['#235789', '#F2B134', '#D5573B'])

    plt.title('Argument Types Supported\n by Social Science Citations per Section')
    plt.xlabel('Section')
    plt.ylabel('Number of Citations')
    plt.savefig(f"{ANALYSIS_DIR}/ss_section_arg_types.png", dpi=300)
    plt.show()


    # get all rows with no section
    to_predict = {}
    get_no_section_query = "SELECT ac.arg_pred, ac.section, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and social_science = 0 and interdisciplinary = 0;"
    

    df = pd.read_sql_query(get_no_section_query, con)
    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.replace({'section': {'introduction': 'Introduction', 'related work': 'Related Work', 'method': 'Method', 'experiments': 'Experiments', 'conclusion': 'Conclusion', 'appendix': 'Appendix'}})
    
    sections = ['Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion', 'Appendix']

    mapping = {s: i for i, s in enumerate(sections)}
    key = df['section'].map(mapping)

    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})

    df = df.groupby(['section','arg_pred']).size().unstack().reindex(sections)\
                                        .plot(kind='bar',color=['#235789', '#F2B134', '#D5573B'])

    plt.title('In-Discipline Citations')
    plt.xlabel('Section')
    plt.ylabel('Number of Citations')
    plt.savefig(f"{ANALYSIS_DIR}/id_section_arg_types.png", dpi=300)
    plt.show()

    get_no_section_query = "SELECT ac.arg_pred, ac.section, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO' and interdisciplinary = 1;"
    

    df = pd.read_sql_query(get_no_section_query, con)
    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.replace({'section': {'introduction': 'Introduction', 'related work': 'Related Work', 'method': 'Method', 'experiments': 'Experiments', 'conclusion': 'Conclusion', 'appendix': 'Appendix'}})
    sections = ['Introduction', 'Related Work', 'Method', 'Experiments', 'Conclusion', 'Appendix']
    mapping = {s: i for i, s in enumerate(sections)}
    key = df['section'].map(mapping)

    df = df.replace({'arg_pred': {'background_claim': 'Background Claim', 'data': 'Data', 'own_claim': 'Own Claim'}})
    df = df.groupby(['section','arg_pred']).size().unstack().reindex(sections)\
                                        .plot(kind='bar',color=['#235789', '#F2B134', '#D5573B'])

    plt.title('Out-of-Discipline Citations')
    plt.xlabel('Section')
    plt.ylabel('Number of Citations')
    plt.savefig(f"{ANALYSIS_DIR}/ood_section_arg_types.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()