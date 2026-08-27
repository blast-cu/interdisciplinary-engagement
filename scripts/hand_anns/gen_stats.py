import pandas as pd
import sqlite3

def main():
    df0 = pd.read_csv("data/manual_anns/articles.csv")
    df1 = pd.read_csv("data/manual_anns/articles1.csv")
    article_df = pd.concat([df0, df1], ignore_index=True)

    citations_df = pd.read_csv("data/manual_anns/citations.csv")
    master_df = pd.read_csv("data/manual_anns/master_citations.csv")
    master_df = master_df.drop_duplicates(subset=["citation_id"], keep="first")
    merged_citations_df = citations_df.merge(master_df, how="left", on="citation_id")
    # print(citations_df.columns)
    print(set(merged_citations_df["category_x"].tolist()))

    if len(citations_df) != len(merged_citations_df):
        print(f"something went wrong with the merge: original df len {len(citations_df)}, new len {len(merged_citations_df)}")
    
    labelled_df = merged_citations_df[merged_citations_df["category_x"].notna()]
    print(set(labelled_df["category_x"].tolist()))
    grouped_df = labelled_df[["citation_id", "interdisciplinary", "category_x"]]
    grouped_df = grouped_df.groupby(["interdisciplinary", "category_x"]).count()
    # grouped_df = grouped_df.groupby(["category_x"]).count()

    print(grouped_df)

    labelled_df = merged_citations_df[merged_citations_df["category_x"].notna()]
    # print(set(labelled_df["category_x"].tolist()))
    # print(labelled_df.columns)
    grouped_df = labelled_df[["citation_id", "section_y", "interdisciplinary"]]
    grouped_df = grouped_df.groupby(["section_y", "interdisciplinary"]).count()
    print(grouped_df)






if __name__ == "__main__":
    main()