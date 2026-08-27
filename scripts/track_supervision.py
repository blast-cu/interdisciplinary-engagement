import argparse
import sqlite3
import os
import json

from data_utils.utils import DATA_DIR, DB_FILENAME
from scripts.visualization.plot_generators import \
    generate_training_stats_venue_plot, generate_training_stats_year_plot


def main(args):

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    # create data_check directory
    data_check_dir = f"{DATA_DIR}data_check"
    os.makedirs(data_check_dir, exist_ok=True)

    # create plots of training stats
    if args.venue:
        generate_training_stats_venue_plot(con, f"{data_check_dir}/track_training_stats_venue.png", processed_only=args.processed_only)
    else:
        generate_training_stats_year_plot(con, f"{data_check_dir}/track_training_stats_year.png", processed_only=args.processed_only)
    

    # check that we used all supervision
    with open(f"{DATA_DIR}models/track_classifier/test_results.json", 'r') as f:
        test_results = json.load(f)

    examples = []
    for split in ["train", "validation", "test"]:
        for example in test_results[split]:
            examples.append(example)

    print(f"Used {len(examples)} training examples to train track classifier")


    check = "SELECT article.year, article.venue, count(article.id) FROM article \
                LEFT JOIN predictions ON predictions.article_id = article.id WHERE article.nlp_css = 1 OR predictions.nlp_css_prediction = 1 \
                GROUP BY article.year, article.venue;"
    
    cur.execute(check)
    check = cur.fetchall()
    for c in check:
        print(c)

    con.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_only", action="store_true", help="Use only processed data")
    parser.add_argument("--venue", action="store_true", help="x axis venue instead of year")

    args = parser.parse_args()
    main(args)