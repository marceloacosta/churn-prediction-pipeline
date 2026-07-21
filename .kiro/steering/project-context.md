---
inclusion: always
---

# Project Context: Churn Prediction Pipeline

## What This Project Is

A standalone, productized churn prediction pipeline on AWS — built as both:
1. A real production ML system (src/churn_pipeline/)
2. A teaching series published on https://buildwithaws.substack.com/ (notebooks/)

The only inputs are: a raw CSV file + a YAML mapping config.
The output is: a scored CSV with churn probabilities, risk tiers, SHAP explanations, and LLM-generated narratives.

## Key Decisions Already Made

- **Algorithm:** XGBoost (SageMaker built-in) — no Docker containers to maintain
- **Orchestration:** SageMaker Pipelines — serverless, native AWS
- **Scoring:** Batch Transform — cheapest option, compute spins up only when needed
- **Explainability:** SHAP + LLM narratives via Amazon Bedrock (Claude)
- **Monitoring:** PSI-based drift detection + MLflow tracking
- **LLM Integration (2 places):**
  1. Auto-mapping: Claude reads CSV columns + sample rows → drafts mapping YAML → human approves
  2. Narrative generation: Claude turns raw SHAP numbers into plain English paragraphs
- **Both LLM steps are non-blocking** — pipeline works without them
- **NOT converting a notebook** — this is built from scratch, standalone
- **NOT using CatBoost** — XGBoost only (SageMaker built-in, no custom containers)

## Writing Style: Feynman-First

Every technical concept MUST be explained by what it DOES before it gets a name. Never introduce jargon without explanation first.

BAD: "Compute AUC-ROC"
GOOD: "Imagine sorting all customers by how likely the model thinks they'll leave. A perfect model puts all actual leavers at the top. AUC measures how close to perfect the sorting is — 1.0 = perfect, 0.5 = random. We call this AUC-ROC and require at least 0.70."

This applies to ALL code docstrings, notebook markdown cells, and documentation.

## Publishing Strategy

- **GitHub repo** (this repo) — source of truth for code + notebooks
- **Jupyter Book** (book/ folder) — rendered on GitHub Pages as a free website
- **Substack** (buildwithaws.substack.com) — one post per phase, links to notebooks

Substack post schedule maps to notebooks:
1. Data Contracts — "Why your ML pipeline breaks when a new client sends data"
2. Schema Validation — "The bouncer that saves your model from garbage data"
3. Feature Engineering — "Your model can't read English — here's how to translate"
4. XGBoost Training — "100 bad guessers that become one great predictor"
5. Evaluation Gates — "How to stop a bad model before it reaches your client"
6. SHAP Explanations — "Your model needs to show its work"
7. Drift Monitoring — "The smoke detector for when the world changes"
8. LLM + ML pipelines — "Using Claude to do the boring parts of ML ops"
9. SageMaker Pipelines — "The factory floor manager for your ML system"

## Repo Structure

```
churn-prediction-pipeline/
├── .kiro/specs/churn-pipeline-aws/    # Spec (requirements.md, design.md, tasks.md)
├── .kiro/steering/                    # This file + any other steering
├── notebooks/                         # Learning path (Jupyter Book chapters)
│   ├── 01_data_contract_and_mapping.ipynb
│   ├── 02_schema_validation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_training_fundamentals.ipynb
│   ├── 05_evaluation_and_scoring.ipynb
│   ├── 06_drift_monitoring.ipynb
│   ├── 07_llm_integration.ipynb
│   └── 08_aws_architecture.ipynb
├── src/churn_pipeline/                # Production code
│   ├── steps/ (validate, features, training, evaluation, scoring, monitoring)
│   └── llm/ (auto_mapping, narrative_generator)
├── configs/                           # Per-client mapping YAMLs
├── tests/ (unit/, property/, integration/)
├── data/                              # Sample datasets or download scripts
├── scripts/run_pipeline.py
├── book/ (_config.yml, _toc.yml)      # Jupyter Book config
├── DEPLOY.md                          # AWS deployment guide
└── .github/workflows/                 # CI
```

## Implementation Approach

Each "task phase" from tasks.md produces BOTH:
1. A notebook (notebooks/0N_*.ipynb) — teaches the concept with Feynman explanations, runs locally
2. Production code (src/churn_pipeline/) — the clean, tested module

The notebook is where you learn and experiment. The src/ code is the production artifact. They reference each other.

AWS-specific code (SageMaker Pipelines, triggers) doesn't fit notebooks well — that goes in src/ only, with an architecture notebook (08) that uses moto mocking to demonstrate concepts.

## Test Datasets

- IBM Telco Customer Churn (7,043 rows) — telecommunications
- d0r1h/customer_churn (37,000 rows) — e-commerce/SaaS
- moaminsharifi/Churn_Modelling (10,000 rows) — banking

## Current Progress

Tasks 1 and 2 are COMPLETE. The following files exist and are implemented:
- src/churn_pipeline/data_contract.py (Task 1.2)
- src/churn_pipeline/mapping_config.py (Task 1.3)
- src/churn_pipeline/steps/validate_data.py (Task 2.1)
- configs/client_telco/mapping.yaml (Task 1.5)
- configs/client_ecommerce/mapping.yaml (Task 1.5)
- configs/client_banking/mapping.yaml (Task 1.5)
- tests/property/test_mapping_roundtrip.py (Task 1.4)
- tests/property/test_schema_validation.py (Task 2.2)
- tests/unit/test_validate_data.py (Task 2.3)
- pyproject.toml with all dependencies (Task 1.1)

## What To Do Next

Resume from Task 3 (Checkpoint — run tests to confirm tasks 1-2 work) → then Task 4 (Feature Engineering).

For each task phase:
1. Write the src/ module (production code with Feynman docstrings)
2. Write the corresponding notebook (teaching narrative + interactive examples)
3. Run property-based tests (if applicable for that phase)
4. Checkpoint — verify everything works before moving on

NOTE: No notebooks have been written yet. When starting notebooks, begin with 01 and 02 covering the already-implemented code (data contract + validation), then continue with new notebooks alongside new src/ code.
