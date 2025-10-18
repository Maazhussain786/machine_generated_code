import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
for model in ["tfidf","xgboost","codebert"]:
    path = f"experiments/{model}_task_a/predictions_test.csv"
    if not pd.io.common.file_exists(path):
        print(model, "no test preds")
        continue
    df = pd.read_csv(path)
    print("\nModel:", model)
    print("Counts y_true:", df['y_true'].value_counts().to_dict())
    print("Counts y_pred:", df['y_pred'].value_counts().to_dict())
    print(classification_report(df['y_true'], df['y_pred'], digits=4))
    print("Confusion matrix:\n", confusion_matrix(df['y_true'], df['y_pred']))
