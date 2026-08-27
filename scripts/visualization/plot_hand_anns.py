import pandas as pd
import sqlite3
from data_utils.utils import DB_FILENAME
from scripts.visualization.plot_generators import COLORS, fix_venue_names

# plot year distribution
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

MATPLOT_STYLE = 'seaborn-v0_8-darkgrid'

def get_year_bucket(year):
        buckets = [(2014, 2016, "1"),
        (2017, 2019, "2"),
        (2020, 2020, "3"),
        (2021, 2021, "4"),
        (2022, 2023, "5"),
        (2024, 2025, "6")]

        for start_year, end_year, num in buckets:
            if (year >= start_year and year <= end_year) or (year == start_year and start_year == end_year):
                if start_year == end_year: return str(start_year)
                else: return f"{start_year} - {end_year}"

            if num == 5:
                print("NO BUCKET")

def main():
    out_path = "data/paper/new_figs"

    a_df = pd.read_csv("data/manual_anns/articles.csv")
    c_df = pd.read_csv("data/manual_anns/citations.csv")
    a_df = fix_venue_names(a_df)

    a_df["year_bucket"] = a_df["year"].apply(get_year_bucket)
    print(a_df)
    c_merged_df = c_df.merge(a_df, how="left", on="abstract")


    matplotlib.style.use(MATPLOT_STYLE)
    rcParams.update({'figure.autolayout': True})

    year_counts = a_df['year_bucket'].value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, color=['#235789'])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Year Distribution of Annotated Articles')
    plt.tight_layout()
    plt.savefig(f'{out_path}/ann_article_year_distribution.png', dpi=300)
    plt.close()

    year_counts = a_df['venue'].value_counts().sort_index()
    
    plt.bar(year_counts.index, year_counts.values, color=COLORS[1:])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Venue Distribution of Annotated Articles')
    plt.tight_layout()
    plt.savefig(f'{out_path}/ann_article_venue_distribution.png', dpi=300)
    plt.close()

    year_counts = c_merged_df['year_bucket'].value_counts().sort_index()
    
    plt.bar(year_counts.index, year_counts.values, color=['#235789'])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Year Distribution of Annotated In-Text Citations')
    plt.tight_layout()
    plt.savefig(f'{out_path}/ann_cit_year_distribution.png', dpi=300)
    plt.close()

    year_counts = a_df['venue'].value_counts().sort_index()
    
    plt.bar(year_counts.index, year_counts.values, color=COLORS[1:])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Venue Distribution of Annotated In-Text Citations')
    plt.tight_layout()
    plt.savefig(f'{out_path}/ann_cit_venue_distribution.png', dpi=300)
    plt.close()


    # filter df to get an example 
    con = sqlite3.connect(DB_FILENAME)
    citation_df = pd.read_sql("SELECT id, interdisciplinary FROM citations", con)
    con.close()
    df = c_merged_df.merge(citation_df, how="left", left_on="citation_id", right_on="id")
    df = df[df["interdisciplinary"] == 1]

    if len(df) == 0:
        print("None left")
        exit()

    # example_df = df[df['category'] == 'substantiation + basis']
    # example_df = example_df[["citation_text", "article_title", "category"]]
    # example_df.to_csv("example.csv")


if __name__ == "__main__":
    main()