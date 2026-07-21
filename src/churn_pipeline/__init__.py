"""
Churn Prediction Pipeline
=========================

A productized churn prediction system that takes raw CSV data and a YAML mapping
config, then produces scored customer lists with churn probabilities, risk tiers,
SHAP explanations, and LLM-generated narratives.

Think of it as an assembly line: raw ingredients (customer data) go in one end,
and a finished report comes out the other — with quality checks at every station.
"""

__version__ = "0.1.0"
