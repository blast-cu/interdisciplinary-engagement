import argparse
import pandas as pd
from sklearn.metrics import classification_report
from data_utils.utils import get_logger, load_file


def main(args):

    SCRIPT_NAME = "get_classification_report"
    logger = get_logger(SCRIPT_NAME)

    # read in results
    results_path = args.results_path
    data = load_file(results_path, logger)
    try:
        test_data = data["test"]  # list of samples
    except KeyError as e:
        logger.error("Input file must be .json with 'test' key and corresponding list of samples")
        raise e

    # collect labels and preds
    y_test = []
    y_pred = []
    for sample in test_data:
        y_test.append(sample["label"])
        y_pred.append(sample["prediction"])
    
    # specify class names
    target_names = []
    if "track" in results_path:
        target_names = target_names
    elif "purpose" in results_path:
        target_names = ["substantiation + basis", "substantiation", "basis", "definition", "analysis", "use", "related work"]
    else:
        logger.warning("Unknown classification task, not providing target names")

    logger.info("Generating report...")
    c_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # save to csv
    c_report_df = pd.DataFrame(c_report).transpose().round(decimals=3)
    c_report_df["support"] = c_report_df["support"].astype(int)
    out_path = results_path.replace(".json", "_classification_report.csv")
    c_report_df.to_csv(out_path)
    logger.info(f"Saved report to {out_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', help='Path to results file')
    args = parser.parse_args()

    main(args)