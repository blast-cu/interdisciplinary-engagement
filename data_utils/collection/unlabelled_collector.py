from data_utils.utils import DATA_DIR
import data_utils.collection.utils as cu
import os
import pandas as pd

"""
Download papers from a list of venues and years.
Create a directory structure for the papers.
Save abstracts to pickle file in /unlabelled directory.
"""


def main():

    df = pd.read_csv(f"{DATA_DIR}csv_data/venue_info.csv")
    for ind in df.index:

        year = df['year'][ind]
        venue = df['venue'][ind]
        url = df['proceedings_url'][ind]

        print("\n -------------------")
        print(f"Processing {venue} {year} papers from {url}")
        print(" -------------------")

        out_dir = f"{DATA_DIR}venues/{venue}/{year}/unlabelled"
        raw_dir = f"{out_dir}/raw/"
        os.makedirs(raw_dir, exist_ok=True)
        
        # download all papers and save abstracts to pickle file.
        download = True
        abstract_dict = cu.download_papers(url, raw_dir, download=download)

        cu.get_abstracts(raw_dir, out_dir, abstract_dict)


if __name__ == "__main__":
    main()