# src/data_loader.py
import os
import pandas as pd
from typing import Tuple

TASK_FILES = {
    "task_a": {
        "train": "task_a_training_set_1.parquet",
        "dev":   "task_a_validation_set.parquet",
        "test":  "task_a_test_set_sample.parquet",
    },
    "task_b": {
        "train": "task_b_training_set.parquet",
        "dev":   "task_b_validation_set.parquet",
        "test":  "task_b_test_set_sample.parquet",
    },
    "task_c": {
        "train": "task_c_training_set_1.parquet",
        "dev":   "task_c_validation_set.parquet",
        "test":  "task_c_test_set_sample.parquet",
    }
}

def load_task_parquet(data_root: str, task: str = "task_a", return_paths: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = task.lower()
    if task not in TASK_FILES:
        raise ValueError(f"Unknown task '{task}'. Valid: {list(TASK_FILES.keys())}")
    task_dir = os.path.join(data_root, task)
    files = TASK_FILES[task]
    train_path = os.path.join(task_dir, files["train"])
    dev_path   = os.path.join(task_dir, files["dev"])
    test_path  = os.path.join(task_dir, files["test"])
    for p in (train_path, dev_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
    train_df = pd.read_parquet(train_path)
    dev_df   = pd.read_parquet(dev_path)
    test_df  = pd.read_parquet(test_path)
    if return_paths:
        return train_df, dev_df, test_df, (train_path, dev_path, test_path)
    return train_df, dev_df, test_df
