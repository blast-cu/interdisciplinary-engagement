import pandas as pd
import sqlite3
from data_utils.utils import DATA_DIR, DB_FILENAME
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import os

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
    get_no_section_query = "SELECT ac.own_claim, ac.data, ac.background_claim, ac.arg_pred, ac.article_id, \
        ac.citation_id, ac.section, citations.field_of_study, citations.social_science, \
        citations.interdisciplinary FROM article_citations AS ac  \
         JOIN citations ON ac.citation_id = citations.id \
            WHERE ac.section IS NOT NULL AND citations.field_of_study IS NOT NULL AND citations.fos_source != 'HABANERO';"

    df = pd.read_sql_query(get_no_section_query, con)
    df['count'] = 1
    df['year'] = df.apply(get_year, axis=1)
    df['venue'] = df.apply(get_conference, axis=1)

    year_plot = df.groupby(['article_id']).agg({'count':'sum', 'own_claim':'sum', 'data':'sum', 'background_claim':'sum', 'year': 'max'})
    year_plot['percent data'] = year_plot['data'] / year_plot['count']
    year_plot['percent own claim'] = year_plot['own_claim'] / year_plot['count']
    year_plot['percent background claim'] = year_plot['background_claim'] / year_plot['count']
    year_plot = year_plot.drop(columns=['count', 'own_claim', 'data', 'background_claim'])
    matplotlib.style.use('seaborn-v0_8-darkgrid')
    rcParams.update({'figure.autolayout': True})
    year_plot = year_plot.groupby(['year']).mean().plot(kind='bar', color=['#235789', '#D5573B', '#F2B134'])
    plt.xlabel('Year')
    plt.ylabel('Average Percentage of Citations')
    plt.title('Percentages of Argument Types Supported by Citations by Year')
    # plt.legend(["Not Interdisciplinary", "Interdisciplinary"])
    
    plt.savefig(f"{ANALYSIS_DIR}/year_arg_types.png", dpi=300)
    plt.show()

    venue_plot = df.groupby('article_id').agg({'venue': 'max', 'count': 'sum', 'own_claim':'sum', 'data':'sum', 'background_claim':'sum'})
    venue_plot['Data'] = venue_plot['data'] / venue_plot['count']
    venue_plot['Own Claim'] = venue_plot['own_claim'] / venue_plot['count']
    venue_plot['Background Claim'] = venue_plot['background_claim'] / venue_plot['count']
    venue_plot = venue_plot.drop(columns=['count', 'own_claim', 'data', 'background_claim'])
    venue_plot = venue_plot.groupby(['venue']).mean().plot(kind='barh', color=['#235789', '#D5573B', '#F2B134'])
    plt.xlabel('Average Percentage of Citations')
    plt.ylabel('Venue')
    plt.title('Percentages of Argument Types Supported by Citations by Venue')
    plt.savefig(f"{ANALYSIS_DIR}/venue_arg_types.png", dpi=300)
    plt.show()

    con.close()


if __name__ == "__main__":
    main()
