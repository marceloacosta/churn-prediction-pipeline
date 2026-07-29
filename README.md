# Churn Prediction Pipeline — From Messy CSV to Production on AWS

A hands-on course that builds a complete churn prediction system step by step. Start with a dirty dataset, end with a production ML pipeline on SageMaker — with LLMs where they make sense.

> **🚧 Course in progress.** Modules 1–7 are available now. Modules 8–10 (SageMaker deployment, MLflow tracking, end-to-end integration) are coming soon. Follow the repo or [subscribe on Substack](https://buildwithaws.substack.com/) to get notified when new modules drop.

## How to Use This Course

1. Click the **Open in Colab** badge next to the module you want.
2. In Colab, go to **File → Save a copy in Drive** immediately.
3. Work in YOUR copy. The original is read-only.
4. Each module is self-contained — it replays previous steps automatically so you can start anywhere.

## Modules

<!-- BADGES:START -->
| # | Module | Status | Open in Colab |
|---|--------|--------|---------------|
| 1 | Data Contracts & Mapping | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/01-data-contracts/01-data-contracts.ipynb) |
| 2 | Schema Validation & Data Cleaning | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/02-schema-validation/02-schema-validation.ipynb) |
| 3 | Feature Engineering | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/03-feature-engineering/03-feature-engineering.ipynb) |
| 4 | Training & Scoring | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/04-training-fundamentals/04-training-fundamentals.ipynb) |
| 5 | Evaluation Gates | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/05-evaluation/05-evaluation.ipynb) |
| 6 | Drift Monitoring | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/06-drift-monitoring/06-drift-monitoring.ipynb) |
| 7 | LLM Integration | ✅ Available | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marceloacosta/churn-prediction-pipeline/blob/main/modules/07-llm-integration/07-llm-integration.ipynb) |
| 8 | SageMaker Pipelines (AWS Deployment) | 🔜 Coming soon | — |
| 9 | MLflow Tracking & Experiment Management | 🔜 Coming soon | — |
| 10 | End-to-End Integration & Production | 🔜 Coming soon | — |
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
- **LLMs:** Amazon Bedrock (Claude) for auto-mapping and narrative generation

## Running Locally

```bash
git clone https://github.com/marceloacosta/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then open notebooks in order from `modules/01-data-contracts/`.

## Repo Structure

```
├── modules/             # Course notebooks (one folder per module)
├── src/churn_pipeline/  # The Python package notebooks import
├── configs/             # Per-client mapping YAMLs
├── data/                # Downloaded at runtime (not committed)
├── pyproject.toml       # Package definition
└── README.md            # This file
```

## Source of Truth

This repo is the source of truth. Your saved Colab copies are yours to edit freely.

---

*Series: [Build with AWS](https://buildwithaws.substack.com/)*
