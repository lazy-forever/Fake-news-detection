import pandas as pd
from sklearn.metrics import roc_auc_score

def calc(rightFileName="./data/right.csv", outputFileName="submit.csv"):
    output=pd.read_csv(outputFileName)
    right=pd.read_csv(rightFileName)

    output_data=output["label"]
    right_data=right["label"]
    return roc_auc_score(right_data,output_data)


if __name__ == "__main__":
    print("aoc:",end='')
    print(calc())