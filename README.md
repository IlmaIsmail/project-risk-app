# Project Risk Prediction — Interpretable Decision Support

An interpretable machine learning framework that predicts project risk from
planning-stage attributes, across five industries, with an explanation for
every prediction.

**Live app:** https://ilma-project-risk-prediction.streamlit.app/

---

## What this does

Most project risk tools detect trouble once a project is already underway,
using execution data such as schedule variance or percentage complete. This
framework predicts risk **before delivery begins**, using only attributes
known at the planning stage — team composition, budget, complexity,
governance maturity, stakeholder structure and delivery context.

Every prediction is accompanied by the factors that drove it, so the output
can be interrogated rather than taken on trust.

## Results

| Metric                                      | Value |
| ------------------------------------------- | ----- |
| Three-class accuracy (Low / Medium / High+) | 78.2% |
| Binary at-risk detection accuracy           | 88.0% |
| Predictions within one risk class of truth  | 99.5% |
| Leave-One-Industry-Out mean AUC             | 0.92  |

Performance is bounded by an irreducible label-noise ceiling in the dataset,
identified and quantified during development rather than left unexplained.
See the thesis for the full analysis.

## Model

A two-layer **stacking ensemble**:

- **Base learners:** XGBoost, LightGBM, Random Forest — cost-sensitive,
  regularised across two rounds
- **Meta-learner:** Logistic Regression, trained on five-fold out-of-fold
  class probabilities from the base learners
- **Features:** 62, derived from 3,203 project records across Information
  Technology, Research & Development, Manufacturing, Marketing and Healthcare

Generalisation is validated by **Leave-One-Industry-Out cross-validation** —
the model is tested on an entire industry it never saw during training,
repeated for all five.

## Explainability

Predictions are explained with **SHAP**, anchored to the High+ class so a
positive contribution always means "increases risk" regardless of which grade
was predicted. During development, SHAP explanations were cross-validated
against **LIME** to confirm that agreement between two independently derived
methods held.

The interface shows the top five contributing factors with the project's own
value for each, flags any value that was auto-filled, and distinguishes
factors a manager can change from those that are largely fixed.

## Running it locally

```bash
git clone https://github.com/YOUR_USERNAME/project-risk-app.git
cd project-risk-app

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

Library versions are pinned exactly, because the serialised model artefacts
are not guaranteed to load identically across major versions.

## Repository structure

```
.
├── app.py                  Streamlit interface
├── requirements.txt        Pinned dependencies
├── runtime.txt             Python version for deployment
├── .streamlit/
│   └── config.toml         Theme and server configuration
└── outputs/
    ├── processed_data.pkl      Encoders, scaler, column schema, lookups
    ├── model_voting.pkl        Base learners and meta-learner
    └── final_feature_names.pkl Expected feature order
```

The app is **schema-driven**: column lists, encoders and the scaler are all
read from `processed_data.pkl` rather than hardcoded, so the interface cannot
drift out of sync with the trained model.

## Important implementation note

Preprocessing must run in this exact order:

```
ordinal-encode  ->  engineer  ->  one-hot  ->  scale  ->  align
```

The engineered features are computed from **ordinal-encoded** values (for
example `3 - Org_Process_Maturity`), so reordering these steps produces
silently invalid predictions. The final `reindex` against the stored column
order guarantees the feature vector matches what the model was trained on.

## Limitations

- Trained on a **synthetic** dataset; results demonstrate methodological
  validity rather than empirical facts about real project outcomes
- Not validated in a live project management setting
- Explanations are **associative, not causal** — they show what correlates
  with elevated risk, not what would change the outcome if altered
- Accuracy is bounded by label noise inherent to the data

## Academic context

Submitted in partial fulfilment of the requirements for the degree of
**Bachelor of Science (Honours) in Information Technology**, Department of
Physical Science, Faculty of Applied Science, University of Vavuniya,
Sri Lanka.

**Author:** Ahamed Ismail Fathima Ilma (2020/ICT/48)

## License

Provided for academic review. Please contact the author before reuse.
