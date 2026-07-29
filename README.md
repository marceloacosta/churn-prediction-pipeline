# Churn Prediction Pipeline — From Messy CSV to Production on AWS

A hands-on course that builds a complete churn prediction system step by step. Start with a dirty dataset, end with a production ML pipeline on SageMaker — with LLMs where they make sense.

## How to Use This Course

1. Click the **Open in Colab** badge next to the module you want.
2. In Colab, go to **File → Save a copy in Drive** immediately.
3. Work in YOUR copy. The original is read-only.
4. Each module picks up where the previous one left off — run them in order.

## Modules

<!-- BADGES:START -->
| # | Module | Open in Colab |
|---|--------|---------------|
| 1 | 01 — Data Contracts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/01-data-contracts/01-data-contracts.ipynb) |
| 2 | 02 — Schema Validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/02-schema-validation/02-schema-validation.ipynb) |
| 3 | 03 — Feature Engineering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/03-feature-engineering/03-feature-engineering.ipynb) |
| 4 | 04 — Training Fundamentals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/04-training-fundamentals/04-training-fundamentals.ipynb) |
| 5 | 05 — Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/05-evaluation/05-evaluation.ipynb) |
| 6 | 06 — Drift Monitoring | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/06-drift-monitoring/06-drift-monitoring.ipynb) |
| 7 | 07 — Llm Integration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/07-llm-integration/07-llm-integration.ipynb) |
<!-- BADGES:END -->

## What You'll Build

By the end of this course, you'll have a working system that:
- Takes any client's raw CSV (with messy, inconsistent data)
- Cleans it (garbage values, duplicates, inconsistent categories)
- Transforms it into model-ready features
- Trains an XGBoost model with proper evaluation gates
- Produces scored customer predictions with risk tiers and explanations
- Monitors for data drift (alerts when the world changes)
- Uses LLMs to auto-map column names and generate plain-English explanations
- Runs in production on AWS SageMaker Pipelines

## Tech Stack

- **Python 3.11+** with pandas, scikit-learn, XGBoost, SHAP
- **AWS:** SageMaker Pipelines, S3, Bedrock (Claude), MLflow
- **Testing:** pytest + Hypothesis (property-based testing)
- **LLMs:** Amazon Bedrock (Claude) for auto-mapping and narrative generation

## Running Locally

```bash
git clone https://github.com/marceloacosta/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then open notebooks in order from `modules/01-data-contracts/`.

## Repo Structure

```
├── modules/          # Course notebooks (one folder per module)
├── src/churn_pipeline/  # Production code
├── configs/          # Per-client mapping YAMLs
├── data/             # Small sample datasets
├── tests/            # Unit + property-based tests
└── scripts/          # Utilities (badge generation, validation)
```

## Source of Truth

This repo is the source of truth. Your saved Colab copies are yours to edit freely.

---

*Series: [Build with AWS](https://buildwithaws.substack.com/)*
