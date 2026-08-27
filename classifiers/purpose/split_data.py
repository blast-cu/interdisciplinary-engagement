import pandas as pd
import csv

valid_labels = ["substantiation + basis", "substantiation", "basis", "definition", "analysis", "use", "related work"]



def main():
    
    input_path = "/Users/alexleto/projects/cross_disciplinary_engagement/data/ann_data/"
    df = pd.read_csv(f"{input_path}all_citations.csv")

    article_ids = df["article_id"].unique()
    print(len(article_ids))

    from sklearn.model_selection import train_test_split

    train_ids, test_ids = train_test_split(article_ids, test_size=0.20, random_state=14)

    train_df = df[df["article_id"].isin(train_ids)]
    test_df = df[df["article_id"].isin(test_ids)]

    print(f"Train: {len(train_df)} citations ({len(train_ids)} articles)")
    print(f"Test:  {len(test_df)} citations ({len(test_ids)} articles)")

    print(train_df.groupby("category").count())
    print(test_df.groupby("category").count())

    train_df.to_csv(f"{input_path}/citations.csv")
    test_df.to_csv(f"{input_path}/test_citations.csv")





  

if __name__ == "__main__":
    main()
