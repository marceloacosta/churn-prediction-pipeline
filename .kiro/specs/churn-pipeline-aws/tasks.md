# Implementation Plan: Productized Churn Prediction Pipeline

## Overview

This plan implements a standalone churn prediction pipeline on AWS from scratch. The only inputs are a raw CSV file and a YAML mapping config. Implementation uses Python with SageMaker SDK, organized into discrete components that can be developed and tested incrementally.

**Implementation language:** Python 3.11+
**Testing frameworks:** pytest + Hypothesis (property-based testing)
**AWS mocking:** moto + custom Bedrock mock
**Infrastructure:** SageMaker Pipelines, S3, MLflow, SNS, Amazon Bedrock
**Property-based testing library:** Hypothesis

## Deliverable Per Task Phase

Each task phase produces TWO outputs:
1. **Production code** in `src/churn_pipeline/` — the tested, importable module
2. **Teaching notebook** in `notebooks/` — Feynman-style explanation + interactive examples that demonstrate the module

The notebook is where the reader learns and experiments. The src/ code is the production artifact. They reference each other. The notebooks are also the basis for Substack posts at https://buildwithaws.substack.com/

Notebook mapping:
- Tasks 1-2 → `notebooks/01_data_contract_and_mapping.ipynb` + `notebooks/02_schema_validation.ipynb`
- Task 4 → `notebooks/03_feature_engineering.ipynb`
- Task 5 → `notebooks/04_training_fundamentals.ipynb`
- Tasks 6, 10 → `notebooks/05_evaluation_and_scoring.ipynb`
- Task 9 → `notebooks/06_drift_monitoring.ipynb`
- Tasks 12-13 → `notebooks/07_llm_integration.ipynb`
- Tasks 15-16 → `notebooks/08_aws_architecture.ipynb`

## Tasks

- [x] 1. Project scaffolding and core data structures
  - [x] 1.1 Create project directory structure and dependencies
    - Create directories: `src/churn_pipeline/`, `src/churn_pipeline/steps/`, `src/churn_pipeline/llm/`, `configs/`, `tests/`, `tests/property/`, `tests/unit/`, `tests/integration/`
    - Create `pyproject.toml` with dependencies: sagemaker, boto3, pandas, numpy, scikit-learn, xgboost, shap, pyyaml, mlflow, hypothesis, pytest, pytest-mock, moto
    - Create `src/churn_pipeline/__init__.py`
    - _Requirements: 9.1_

  - [x] 1.2 Implement data contract schema definition
    - Create `src/churn_pipeline/data_contract.py`
    - Define `Tier` enum (REQUIRED=1, ENGAGEMENT=2, DEMOGRAPHIC=3)
    - Define `FieldSpec` dataclass with: name, dtype, tier, description, allowed_values
    - Define `STANDARD_SCHEMA` dict with all Tier 1/2/3 fields as specified in design
    - Include Feynman-style docstrings explaining what each tier means and why
    - _Requirements: 2.1_

  - [x] 1.3 Implement mapping config parser and serializer
    - Create `src/churn_pipeline/mapping_config.py`
    - Define `MappingConfig` dataclass: client_id, source_description, column_mappings, value_mappings, type_coercions
    - Implement `load_mapping_config(yaml_path) -> MappingConfig` to parse YAML
    - Implement `serialize_mapping_config(config) -> str` to write YAML
    - Implement `apply_mapping(df, config) -> pd.DataFrame` to rename columns, apply value mappings, coerce types
    - Include Feynman-style docstrings explaining the Rosetta Stone concept
    - _Requirements: 2.5, 2.6_

  - [x]* 1.4 Write property test for mapping config round-trip
    - **Property 1: Mapping Config Round-Trip Consistency**
    - Generate random valid MappingConfig objects using Hypothesis (random client_ids, random column_mappings dicts, random value_mappings, random type_coercions)
    - Serialize to YAML string, parse back, verify structural equivalence
    - Minimum 100 iterations
    - **Validates: Requirements 2.5, 2.6**

  - [x] 1.5 Create sample mapping configs for all three test datasets
    - Create `configs/client_telco/mapping.yaml` for IBM Telco dataset (customerID→customer_id, tenure→tenure_months, etc.)
    - Create `configs/client_ecommerce/mapping.yaml` for d0r1h/customer_churn dataset
    - Create `configs/client_banking/mapping.yaml` for moaminsharifi/Churn_Modelling dataset
    - _Requirements: 2.5_

- [x] 2. Schema validation
  - [x] 2.1 Implement dataset validator
    - Create `src/churn_pipeline/steps/validate_data.py`
    - Define `ValidationResult` dataclass: is_valid, tier1_present, tier1_missing, tier2_present, tier2_missing, tier3_present, tier3_missing, errors
    - Implement `validate_dataset(df, schema) -> ValidationResult`
    - Rule: is_valid=False if ANY Tier 1 field missing; is_valid=True if all Tier 1 present (regardless of Tier 2/3)
    - Include Feynman-style docstrings explaining the "bouncer at the door" concept
    - _Requirements: 2.2, 2.3, 2.4_

  - [x]* 2.2 Write property test for schema validation completeness
    - **Property 2: Schema Validation Completeness**
    - Generate random DataFrames with ALL Tier 1 fields + random Tier 2/3 subsets → verify is_valid=True always
    - Generate random DataFrames MISSING at least one Tier 1 field → verify is_valid=False always
    - Minimum 100 iterations
    - **Validates: Requirements 2.2, 2.3, 2.4**

  - [x]* 2.3 Write unit tests for schema validation edge cases
    - Test: empty dataframe → is_valid=False
    - Test: wrong types in Tier 1 field (string where int expected) → error reported
    - Test: exact Tier 1 only → is_valid=True with all Tier 2/3 logged as missing
    - _Requirements: 2.2, 2.3, 2.4_

- [ ] 3. Checkpoint - Core data layer
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Feature engineering
  - [ ] 4.1 Implement feature engineering module
    - Create `src/churn_pipeline/steps/feature_engineering.py`
    - Define `FeatureArtifacts` dataclass: scaler, encoders, impute_values, feature_names
    - Implement `engineer_features(df, fit=True, artifacts=None) -> (np.ndarray, FeatureArtifacts)`
    - Training mode (fit=True): fit new encoders/scalers, compute median/mode imputation values, create interaction features (monthly_charges × tenure_months)
    - Scoring mode (fit=False): apply pre-fitted artifacts without re-fitting
    - Include Feynman-style docstrings explaining "prep chef" concept — why raw data needs chopping before the model can eat it
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 4.2 Write property test for feature matrix integrity
    - **Property 3: Feature Matrix Integrity**
    - Generate random schema-conformant DataFrames (with some nulls, mixed types, varying row counts)
    - Verify output: same row count as input, all float64 columns, zero NaN values
    - Minimum 100 iterations
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [ ]* 4.3 Write property test for feature engineering idempotence
    - **Property 11: Feature Engineering Idempotence (Scoring Mode)**
    - Generate random DataFrames, run in fit mode to get artifacts
    - Run twice in scoring mode with same artifacts, verify np.array_equal on outputs
    - Minimum 100 iterations
    - **Validates: Requirements 3.1, 3.5**

- [ ] 5. Training utilities (splits and class imbalance)
  - [ ] 5.1 Implement stratified splitting and scale_pos_weight
    - Create `src/churn_pipeline/steps/training.py`
    - Implement `create_stratified_splits(features, labels, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)` preserving class proportions
    - Implement `compute_scale_pos_weight(labels) -> float` returning count(0s)/count(1s)
    - Define `HYPERPARAMETER_RANGES` dict with all XGBoost tuning ranges
    - Include Feynman-style docstrings explaining why stratification matters and what scale_pos_weight does
    - _Requirements: 4.4, 4.5_

  - [ ]* 5.2 Write property tests for stratified split and scale_pos_weight
    - **Property 4: Stratified Split Class Preservation**
    - Generate random binary label arrays, split, verify each split's class proportion is within 5pp of original
    - **Property 5: Scale_Pos_Weight Computation Correctness**
    - Generate random binary arrays with minority <30%, verify weight = count(0s)/count(1s)
    - Minimum 100 iterations each
    - **Validates: Requirements 4.4, 4.5**

- [ ] 6. Scoring utilities (risk tier + SHAP extraction)
  - [ ] 6.1 Implement risk tier assignment and SHAP extraction
    - Create `src/churn_pipeline/steps/scoring.py`
    - Implement `assign_risk_tier(probability: float) -> str` with thresholds: high >= 0.7, medium 0.4-0.7, low < 0.4
    - Implement `extract_top_reasons(shap_values, feature_names, top_n=3) -> List[str]` sorting by absolute SHAP value
    - Implement `format_predictions(customer_ids, probabilities, shap_values, feature_names, narratives) -> pd.DataFrame` producing final output CSV schema
    - Include Feynman-style docstrings explaining risk tiers and "show your work" concept
    - _Requirements: 6.2, 6.3, 6.4_

  - [ ]* 6.2 Write property tests for risk tier and SHAP
    - **Property 7: Risk Tier Assignment Consistency and Monotonicity**
    - Generate random floats in [0.0, 1.0], verify deterministic tier mapping
    - Generate random pairs (p1 > p2), verify tier(p1) >= tier(p2)
    - **Property 8: SHAP Explanation Completeness**
    - Generate random SHAP arrays (length >= 3) and feature name lists, verify output always has exactly 3 reasons from the feature list
    - Minimum 100 iterations each
    - **Validates: Requirements 6.3, 6.4**

- [ ] 7. Client data isolation
  - [ ] 7.1 Implement S3 path manager
    - Create `src/churn_pipeline/s3_paths.py`
    - Implement `S3PathManager` class generating all paths for a client_id: raw/, configs/, processed/, models/, outputs/, monitoring/
    - Implement `validate_path_belongs_to_client(path, client_id) -> bool`
    - Include Feynman-style docstrings explaining the "locked floor in a shared building" concept
    - _Requirements: 10.1, 10.3_

  - [ ]* 7.2 Write property test for client data isolation
    - **Property 9: Client Data Isolation**
    - Generate random client_id strings, call all path generation methods, verify every path contains the client_id
    - Verify validation function returns True for correct client, False for different random client_id
    - Minimum 100 iterations
    - **Validates: Requirements 10.1, 10.3, 10.4**

- [ ] 8. Checkpoint - Core logic complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Drift monitoring
  - [ ] 9.1 Implement PSI computation and drift detection
    - Create `src/churn_pipeline/steps/monitoring.py`
    - Define `DriftReport` dataclass: run_date, features_checked, features_drifted, psi_scores, alert_triggered
    - Implement `compute_psi(reference, current, bins=10) -> float` using the bucket-divergence formula
    - Implement `check_drift(training_stats, current_data, threshold=0.2) -> DriftReport`
    - Include Feynman-style docstrings explaining PSI step-by-step (divide into buckets, compare percentages, compute divergence)
    - _Requirements: 8.1, 8.2_

  - [ ]* 9.2 Write property test for PSI symmetry baseline
    - **Property 10: PSI Symmetry Baseline**
    - Generate random float arrays (100+ elements), compute PSI against self, verify result ≈ 0.0 (within 1e-10)
    - Minimum 100 iterations
    - **Validates: Requirements 8.1**

  - [ ]* 9.3 Write unit tests for drift detection
    - Test: identical distributions → PSI = 0, alert_triggered = False
    - Test: dramatically shifted distributions → PSI > 0.2, alert_triggered = True
    - Test: PSI = 0.19 → no alert, PSI = 0.21 → alert (boundary)
    - _Requirements: 8.1, 8.2_

- [ ] 10. Model evaluation gate
  - [ ] 10.1 Implement evaluation module
    - Create `src/churn_pipeline/steps/evaluation.py`
    - Define `EvaluationResult` dataclass: passed, auc_roc, f1_score, precision, recall, threshold
    - Implement `evaluate_model(y_true, y_pred_proba, min_auc=0.70) -> EvaluationResult`
    - Implement `generate_model_card(eval_result, training_params, dataset_info, feature_list) -> Dict`
    - Include Feynman-style docstrings explaining AUC (sorting quality), F1 (balance of mistakes), and the gate concept
    - _Requirements: 5.1, 5.2, 5.4_

  - [ ]* 10.2 Write property test for evaluation gate determinism
    - **Property 6: Evaluation Gate Determinism**
    - Generate random metric tuples (auc, f1, precision, recall all in [0,1])
    - Verify pass/fail depends ONLY on auc >= 0.70 regardless of other metrics
    - Minimum 100 iterations
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 10.3 Write unit tests for evaluation edge cases
    - Test: AUC exactly 0.70 → passed=True
    - Test: AUC 0.699 → passed=False
    - Test: model card contains all required keys (model_id, metrics, hyperparameters, etc.)
    - _Requirements: 5.1, 5.2, 5.4_

- [ ] 11. Checkpoint - All core components implemented
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. LLM Auto-Mapping module
  - [ ] 12.1 Implement auto-mapping with Bedrock
    - Create `src/churn_pipeline/llm/auto_mapping.py`
    - Implement `read_csv_metadata(s3_path, sample_size=5) -> (columns, sample_rows)` to read column names and sample data
    - Implement `build_mapping_prompt(columns, sample_rows, standard_fields) -> str` constructing the Claude prompt with data contract fields and client column names
    - Implement `call_bedrock_for_mapping(prompt) -> Optional[List[ColumnMapping]]` calling Bedrock and parsing the response
    - Implement `write_draft_yaml(mappings, client_id, output_path) -> str` writing .draft.yaml with confidence scores
    - Implement `is_mapping_approved(config_path) -> bool` checking if mapping.yaml (not .draft.yaml) exists
    - Include Feynman-style docstrings explaining why this is a natural language problem
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 12.2 Write unit tests for auto-mapping
    - Test: prompt construction includes all standard field names and sample data
    - Test: draft YAML includes confidence scores for each mapping
    - Test: is_mapping_approved returns False for .draft.yaml, True for mapping.yaml
    - Test: Bedrock failure returns None and doesn't raise
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6_

- [ ] 13. LLM Narrative Generator module
  - [ ] 13.1 Implement narrative generation with Bedrock
    - Create `src/churn_pipeline/llm/narrative_generator.py`
    - Define `NarrativeRequest` dataclass: customer_id, churn_probability, risk_tier, top_shap_features
    - Define `NarrativeResult` dataclass: customer_id, narrative, success
    - Implement `build_narrative_prompt(batch, feature_definitions) -> str` batching multiple customers into one prompt
    - Implement `call_bedrock_for_narratives(prompt) -> Optional[Dict[str, str]]` calling Bedrock for batch narratives
    - Implement `parse_narrative_response(response_text, expected_ids) -> Dict[str, str]` matching narratives to customer IDs
    - Implement `generate_narratives_for_batch(customers, batch_size=50) -> Dict[str, NarrativeResult]`
    - Include Feynman-style docstrings explaining why raw SHAP numbers need translation to English
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 13.2 Write property test for narrative prompt completeness
    - **Property 12: Narrative Prompt Completeness**
    - Generate random batches of NarrativeRequest objects (with random customer IDs, probabilities, tiers, SHAP features)
    - Verify prompt string contains every customer's ID, probability, tier, and all SHAP feature names
    - Minimum 100 iterations
    - **Validates: Requirements 7.3**

  - [ ]* 13.3 Write unit tests for narrative module
    - Test: LLM failure sets narrative = "N/A" and success = False
    - Test: system prompt includes "non-technical language" and "under 150 words"
    - Test: batch of 50 customers produces single prompt (not 50 separate calls)
    - Test: response parsing correctly matches narratives to customer IDs
    - _Requirements: 7.4, 7.5, 7.6_

- [ ] 14. Checkpoint - LLM modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. SageMaker Pipeline integration
  - [ ] 15.1 Implement SageMaker Processing step wrappers
    - Create `src/churn_pipeline/sagemaker_steps.py`
    - Implement wrapper functions packaging each processing step (validate, feature_engineering, evaluate, shap, narratives, monitor) as SageMaker ProcessingStep objects
    - Each wrapper accepts: input S3 paths, output S3 paths, client_id, pipeline role ARN
    - Include Feynman-style docstrings explaining what a ProcessingStep is (a container that runs one Python script)
    - _Requirements: 9.1_

  - [ ] 15.2 Implement training and transform step wrappers
    - Implement `create_training_step()` configuring XGBoost Estimator with hyperparameter ranges and creating TuningStep
    - Implement `create_transform_step()` configuring Batch Transform for scoring
    - Implement conditional model registration (ConditionStep: only register if AUC >= 0.70)
    - Include Feynman-style docstrings explaining the tuning step ("trying all oven temperatures") and Batch Transform ("hire a temp worker")
    - _Requirements: 4.1, 4.2, 6.1, 9.1_

  - [ ] 15.3 Implement pipeline definition and wiring
    - Create `src/churn_pipeline/pipeline.py`
    - Implement `create_churn_pipeline(client_id, role, bucket) -> Pipeline` wiring all steps in order with dependencies
    - Pipeline parameters: client_id, min_auc_threshold, psi_threshold, run_date, narrative_batch_size
    - Add FailStep + SNS notification on any step failure
    - Include Feynman-style docstrings explaining how the "assembly line manager" connects everything
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ] 15.4 Implement pipeline triggers
    - Create `src/churn_pipeline/triggers.py`
    - Implement scheduled trigger (EventBridge rule for weekly cron)
    - Implement S3 event trigger (new file in client raw/ prefix)
    - Implement manual trigger (boto3 start_pipeline_execution call)
    - _Requirements: 9.5_

- [ ] 16. MLflow integration
  - [ ] 16.1 Implement MLflow tracking wrapper
    - Create `src/churn_pipeline/tracking.py`
    - Implement `log_training_run(params, metrics, artifacts_path, client_id)` logging training experiments
    - Implement `log_pipeline_execution(client_id, step_durations, output_path, drift_report)` logging pipeline runs
    - Implement `log_drift_report(drift_report, client_id)` logging monitoring results
    - Include Feynman-style docstrings explaining why tracking matters ("the lab notebook")
    - _Requirements: 4.3, 8.3, 9.4_

- [ ] 17. Checkpoint - Pipeline fully wired
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. End-to-end validation
  - [ ] 18.1 Create pipeline execution script
    - Create `scripts/run_pipeline.py` accepting client_id as argument
    - Include dry-run mode that validates config without launching SageMaker jobs
    - Include Feynman-style comments explaining each step of execution
    - _Requirements: 9.1, 9.3_

  - [ ]* 18.2 Write integration tests for end-to-end flow
    - Test full pipeline with IBM Telco dataset (100-row subset) using moto for S3 mocking
    - Verify output CSV schema: customer_id, churn_probability, risk_tier, top_3_reasons, narrative_explanation
    - Verify drift detection on synthetic drift injection
    - Verify auto-mapping produces valid draft YAML for IBM Telco columns
    - _Requirements: 2.2, 3.1, 6.2, 8.1, 1.3_

  - [ ] 18.3 Validate cross-dataset compatibility
    - Run validation + feature engineering on all 3 test datasets using their mapping configs
    - Verify all datasets pass schema validation with their respective Tier coverage
    - Document which Tier 2/3 fields each dataset provides
    - _Requirements: 2.2, 2.5_

- [ ] 19. Final checkpoint - All tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate 12 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Feynman-style docstrings are REQUIRED in ALL implementation code — every concept explained before named
- Implementation language: Python 3.11+ with type hints throughout
- All SageMaker interactions use the SageMaker Python SDK v2
- LLM steps (auto-mapping + narratives) are non-blocking — pipeline works without them
- LLM calls use Amazon Bedrock (Claude) via boto3 bedrock-runtime client
