import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Projects\Final_project\results\results_task_a_20251018_180521.csv")

df = df[df['split'] == 'test']

plt.figure(figsize=(8,5))
plt.bar(df['model'], df['accuracy'], label='Accuracy', alpha=0.6)
plt.bar(df['model'], df['f1_macro'], label='F1 Macro', alpha=0.6)
plt.title('Model Performance Comparison (Test Set)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(r"D:\Projects\Final_project\plots\comparison_clean.png")
plt.show()
