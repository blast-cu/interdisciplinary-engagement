import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_env

def main():
    env = load_env()
    # read in the csv of predictions
    df = pd.read_csv(f"{env['FINAL_PRED_PATH']}/citations_with_purpose_10.csv")
    print(df.head())
    print(len(df))


if __name__ == "__main__":
    main()