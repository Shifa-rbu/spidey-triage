# Spider-Man Incident Triage

Predicts `response_priority` (Low / Medium / High) for Spider-Man incidents. Built for The Code Breakers recruitment task (2nd-year track).

**Colab notebook:** https://colab.research.google.com/drive/1mkpJPsqmvjwlcMIw5oYHLOKmiClxinZ2?usp=sharing

**Idea:** in an emergency, calling a High incident "Medium" is worse than over-flagging a Medium one. Models are therefore judged on macro-F1, High-class recall and a simple triage cost, not only accuracy (the data is dominated by Medium).

## What is inside
| File | Purpose |
|---|---|
| `notebook.ipynb` | Full pipeline (same code as `train.py`) |
| `train.py` | Script version: audit, split, baselines, 6 models, tuning, test evaluation, plots |
| `app.py` | Gradio web app for live testing (bonus) |
| `requirements.txt` | Dependencies |
| `outputs/` | Plots, tables and final metrics written by the run |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset at `data/spiderman_incident_priority_dataset.csv`. If the file is missing, the code downloads it from the public Google Sheet.

## Run
```bash
python train.py        # trains, evaluates on the test set, saves model.joblib and outputs/
python app.py          # opens the live demo at http://127.0.0.1:7860
```

## Method
- Stratified 80/20 train/test split, `random_state=42`. The test set is used once, at the end.
- Everything (imputation, scaling, encoding) sits inside scikit-learn Pipelines, so nothing leaks from validation or test data.
- 5-fold stratified CV on the training set to compare models and for `RandomizedSearchCV` (scored on macro-F1).
- Models: Dummy, Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting, MLP. Logistic Regression, Random Forest and Gradient Boosting are tuned. The final model is chosen on CV score, never on test score.
- Extras: triage-cost metric (cost numbers are my own assumption) and a with/without `response_delay_minutes` check.

## Final metrics (held-out test set)
<!-- METRICS_START -->
**Final model:** Logistic Regression (tuned) (seed 42, held-out test set)

| Model | accuracy | balanced acc | macro precision | macro recall | macro F1 | High recall | ROC-AUC (ovr) | triage cost (lower=better) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.869 | 0.829 | 0.819 | 0.829 | 0.823 | 0.740 | 0.960 | 0.369 |
| Logistic Regression | 0.841 | 0.870 | 0.767 | 0.870 | 0.806 | 0.860 | 0.956 | 0.397 |
| MLP (neural net) | 0.822 | 0.753 | 0.762 | 0.753 | 0.754 | 0.580 | 0.935 | 0.541 |
| Hist Gradient Boosting (tuned) | 0.806 | 0.785 | 0.728 | 0.785 | 0.751 | 0.640 | 0.921 | 0.538 |
| Hist Gradient Boosting | 0.791 | 0.696 | 0.722 | 0.696 | 0.699 | 0.440 | 0.917 | 0.672 |
| Random Forest (tuned) | 0.756 | 0.696 | 0.689 | 0.696 | 0.677 | 0.500 | 0.902 | 0.762 |
| Random Forest | 0.775 | 0.539 | 0.816 | 0.539 | 0.581 | 0.180 | 0.909 | 0.775 |
| Decision Tree | 0.612 | 0.658 | 0.544 | 0.658 | 0.561 | 0.480 | 0.800 | 1.097 |
| Dummy (always Medium) | 0.703 | 0.333 | 0.234 | 0.333 | 0.275 | 0.000 | 0.500 | 0.922 |
<!-- METRICS_END -->

## Justification
Written in the notebook's last cell.
