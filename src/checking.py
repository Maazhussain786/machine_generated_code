import pandas as pd
from src.data_loader import load_task_parquet
train_df, dev_df, test_df = load_task_parquet(r"D:\Projects\Final_project\data", "task_a")
print("Train labels:", train_df['label'].value_counts(normalize=False).to_dict())
print("Dev   labels:", dev_df['label'].value_counts(normalize=False).to_dict())
print("Test  labels:", test_df['label'].value_counts(normalize=False).to_dict())
# show some examples
print("Test head:\n", test_df[['label','code']].head(5).to_string(index=False))
