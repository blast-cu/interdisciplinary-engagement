import os
import sqlite3
import pickle
import pandas as pd
import requests
import json
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from data_utils.utils import DATA_DIR, DB_FILENAME

con = sqlite3.connect(DB_FILENAME)
cur = con.cursor()

def get_year(row):
    cur.execute(f"SELECT year FROM article WHERE id = {row['article_id']};")
    return cur.fetchone()[0]

def main():

    # add_ss_col_query = "ALTER TABLE citations ADD COLUMN social_science BOOLEAN"
    # cur.execute(add_ss_col_query)

    # add_ss_col_query = "ALTER TABLE citations ADD COLUMN interdisciplinary BOOLEAN"
    # cur.execute(add_ss_col_query)

    f = open("data/labelled_fos.csv", "r")
    df = pd.read_csv(f)
    f.close()
    df = df.fillna(0)
    fields = df["field_of_study"].tolist()
    social_science = df["SS"].tolist()
    interdisciplinary = df["CD"].tolist()

    field_dict = {}
    for i in range(len(fields)):
        ss = int(social_science[i])
        cd = int(interdisciplinary[i])
        field_dict[fields[i]] = {}
        field_dict[fields[i]]["ss"] = ss
        field_dict[fields[i]]["cd"] = cd


    # get rows to iterate over
    fos_query = "SELECT article_citations.article_id, citations.field_of_study FROM article_citations\
                    JOIN citations ON article_citations.citation_id = citations.id\
                        WHERE citations.field_of_study IS NOT NULL and article_citations.section is not null and citations.social_science=1;"
    
    df = pd.read_sql_query(fos_query, con)
    df['year'] = df.apply(get_year, axis=1)
    # print(df)
    fos = df['field_of_study'].tolist()
    fos_clean = []
    for f in fos:
        f = f.lower().strip()
        if ";" in f:
            f_list = f.split(";")
            for f in f_list:
                f = f.strip()
                fos_clean.append(f)
        else:
            fos_clean.append(f)

    get_max = {}
    for f in fos_clean:
        if field_dict[f]["ss"] == 0:
            continue
        if f in get_max:
            get_max[f] += 1
        else:
            get_max[f] = 1
    get_max = {k: v for k, v in sorted(get_max.items(), key=lambda item: item[1], reverse=True)}
    
    plot = {}
    plot['Field'] = []
    plot['Number of Citations'] = []

    for k, v in get_max.items():
        plot['Field'].append(k)
        plot['Number of Citations'].append(v)

    plot['Field'] = plot['Field'][:10]
    plot['Number of Citations'] = plot['Number of Citations'][:10]
    matplotlib.style.use('seaborn-v0_8-darkgrid')
    rcParams.update({'figure.autolayout': True})
    plot_df = pd.DataFrame(plot)
    plot_df.plot(kind='bar', x='Field', y='Number of Citations', color=['#235789'])
    plt.legend().remove()
    plt.xlabel('Field of Study')
    plt.ylabel('Number of Citations')
    plt.title('Top 10 Social Science Fields of Study')
    plt.savefig(f"{DATA_DIR}/analysis/top_10_ss_fos.png", dpi=300)
    plt.show()
    

    
    con.close()

if __name__ == "__main__":
    main()