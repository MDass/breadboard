import csv
import pandas as pd
from sklearn.model_selection import train_test_split

test_ratio = 0.2
dataset_path = "create_finetuned/dataset.csv"
train_path = "train.csv"
test_path = "test.csv"

df = pd.read_csv(dataset_path)
train_df, test_df = train_test_split(df, test_size=test_ratio)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)



