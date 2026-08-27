import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

YEAR_BUCKETS = [
        (2015, 2016, "2015-2016"),
        (2017, 2019, "2017-2019"),
        (2020, 2020, "2020"),
        (2021, 2021, "2021"),
        (2022, 2023, "2022-2023"),
        (2024, 2025, "2024-2025")
    ]

def assign_year_bucket(year):
    for start, end, label in YEAR_BUCKETS:
        if start <= year <= end:
            return label
    return np.nan  # for years outside all buckets

def get_category_distribution(df, col):
    counts = (
    df.groupby([col, "category"])
      .size()
      .reset_index(name="count")
    )
    table = (
    counts.pivot(index=col, columns="category", values="count")
          .fillna(0)
          .astype(int)
    )

    return table

def create_plots(table, fig_path, xlabel, ylabel, title, legend_title, plot_list = ['stacked', 'percent_stacked', 'line', 'heatmap']):
    MATPLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLORS = ['#92B4A7', '#D5573B', '#FFC15E', '#A9C7EE', '#26532B', '#FDA172', '#235789']
    
    mpl.style.use(MATPLOT_STYLE)
    mpl.rcParams.update({'figure.autolayout': True})
    # df = df.plot(kind='bar', y='percent pos', color=['#235789'])

    # add labels and save.
    # plt.title('Percentage of NLP Publications Identified as NLP+CSS by Year')
    # plt.legend().remove()
    # plt.xlabel('Year')
    # plt.ylabel('Percentage of Papers')
    # plt.tight_layout()
    # plt.savefig(filename, dpi=300)

    if 'stacked' in plot_list:
        fig, ax = plt.subplots()
        
        table.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=COLORS
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(
            title="Category",
            fontsize=9,
            title_fontsize=10,
            frameon=False,
            labelspacing=0.4,
            handlelength=1.2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()
        plt.savefig(fig_path + "stacked_bar.png", dpi=300)

    if 'percent_stacked' in plot_list:
        pct_table = table.div(table.sum(axis=1), axis=0)

        fig, ax = plt.subplots()
        pct_table.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            edgecolor="black"
        )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(0, 1)

        ax.yaxis.set_major_formatter(
            mpl.ticker.PercentFormatter(1.0)
        )

        ax.legend(title=legend_title, frameon=False)

        plt.tight_layout()
        plt.savefig(fig_path + "percentage_stacked_bar.svg")

    if 'line' in plot_list:
        fig, ax = plt.subplots()
        for col in table.columns:
            ax.plot(
                table.index,
                table[col],
                marker="o",
                label=col
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title=legend_title, frameon=False)

        plt.tight_layout()
        plt.savefig(fig_path + "line_chart.svg")

    if 'heatmap' in plot_list:
        # heatmap
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)
        plt.savefig(fig_path + "heatmap.svg")

if __name__ == '__main__':
    file_path = 'data/manual_anns/citations.csv'

    df = pd.read_csv(file_path)
    df = df[df['category'].notna()]

    articles = pd.read_csv('data/manual_anns/articles.csv')
    articles = articles[['id', 'year', 'venue']]
    df = df.merge(articles, left_on='article_id', right_on='id', how = 'left')

    df["year_bucket"] = df["year"].apply(assign_year_bucket)
    table = get_category_distribution(df, 'year_bucket')
    fig_path = 'data/paper/figures/purpose_bucket_distributions/'
    os.makedirs(fig_path, exist_ok=True)

    create_plots(
        table=table,
        fig_path=fig_path,
        xlabel='Year Bucket',
        ylabel='Category',
        title='Distribution of Purpose Annotations by Year',
        legend_title='Category Count',
        plot_list=['stacked']
    )

    table = get_category_distribution(df, 'venue')
    fig_path = 'data/paper/figures/purpose_venue_distributions/'
    os.makedirs(fig_path, exist_ok=True)

    create_plots(
        table=table,
        fig_path=fig_path,
        xlabel='Venue',
        ylabel='Category Count',
        title='Distribution of Purpose Annotations by Venue',
        legend_title='Category',
        plot_list=['stacked']
    )
