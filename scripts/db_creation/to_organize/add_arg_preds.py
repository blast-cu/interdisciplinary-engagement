import json
import sqlite3
from data_utils.utils import DATA_DIR, DB_FILENAME

def main():
    f = open("data/arg_predictions.json")
    data = json.load(f)
    f.close()
    # print(data)

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()

    add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN arg_pred TEXT;"
    cur.execute(add_ss_col_query)
    add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN own_claim BOOLEAN;"
    cur.execute(add_ss_col_query)

    add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN background_claim BOOLEAN;"
    cur.execute(add_ss_col_query)

    add_ss_col_query = "ALTER TABLE article_citations ADD COLUMN data BOOLEAN;"
    cur.execute(add_ss_col_query)

    for id, pred in data.items():
        article_id = id.split("_")[0]
        citation_id = id.split("_")[1]
        pred = pred['pred']
        add_query = f"UPDATE article_citations SET arg_pred = '{pred}' \
            WHERE citation_id = {citation_id} and article_id = {article_id};"
        # print(add_query)
        cur.execute(add_query)


        add_query = f"UPDATE article_citations SET {pred} = 1 \
            WHERE citation_id = {citation_id} and article_id = {article_id};"
        print(add_query)
        cur.execute(add_query)

    con.commit()
    con.close()

if __name__ == "__main__":
    main()