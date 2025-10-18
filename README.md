# 🧠 AI vs Human Code Detection — Baseline Pipeline

This repository contains the complete **training, evaluation, and visualization pipeline** for detecting whether a code snippet is **AI-generated** or **human-written**.  
It accompanies the [AI vs Human Code Detection Dataset](https://huggingface.co/datasets/mhb-maaz/ai-detector-dataset) and [baseline models](https://huggingface.co/mhb-maaz/Machine_generated_code_detection).

---

## 📊 Project Overview

- **Task:** Binary classification → AI code vs Human code  
- **Languages:** Python, C++, and more  
- **Models:** TF-IDF, XGBoost (AST features), CodeBERT fine-tune  
- **Dataset size:** Train (500k), Dev (100k), Test (10k balanced)

---

## 📂 Repository Contents

| Folder / File       | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `src/`               | All model training, preprocessing, and evaluation scripts    |
| `plots/`             | Confusion matrices, metric curves, and comparison plots      |
| `results/`           | Evaluation metrics saved as JSON/CSV                         |
| `main.py`            | Complete training + evaluation pipeline                       |
| `requirements.txt`   | Python dependencies                                          |
| `README.md`          | This documentation                                          |

---

## 🧰 Dependencies

```bash
pip install -r requirements.txt
Key libraries used:

scikit-learn

xgboost

transformers

torch

matplotlib / seaborn

pandas / numpy

 Training & Evaluation
# Train all models (TF-IDF, XGBoost, CodeBERT)
python main.py --evaluate_test

# Evaluate models only on test set
python src/evaluate_test_models.py


All predictions and metrics are automatically stored in:

experiments/
plots/
results/


Load dataset:
from datasets import load_dataset
dataset = load_dataset("https://huggingface.co/datasets/mhb-maaz/ai-detector-dataset")


Load model:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("https://huggingface.co/mhb-maaz/Machine_generated_code_detection")


📊 Example Results (10k Balanced Test Set)
Model	Accuracy	F1 Macro	Precision	Recall
TF-IDF	0.51	0.37	0.53	0.51
XGBoost	0.94	0.94	0.95	0.94
CodeBERT	0.99	0.99	0.99	0.99

🖼 Confusion matrices and plots are saved in plots/.

📌 Contributors

Maaz Hussain

Muhammad Abdul Daym

Hamza Iqbal

Bilal Atif

📜 License

MIT License © 2025
Authors: Maaz Hussain, Muhammad Abdul Daym, Hamza Iqbal, Bilal Atif

📌 Acknowledgments

This project was developed as part of CS-272: Artificial Intelligence course at NUST.

✉️ Maintainer: Maaz Hussain — Hugging Face


---
