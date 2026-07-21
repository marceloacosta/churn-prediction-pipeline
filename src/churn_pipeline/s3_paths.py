"""
S3 Path Manager — The Locked Floor in a Shared Building
========================================================

Imagine a shared office building where each company has their own locked floor.
They're in the same building (AWS account), share the same elevators (services),
but no one can access another company's floor (data).

Our pipeline runs in a single AWS account serving multiple clients. Each client's
data lives under its own S3 prefix — like a separate filing cabinet with its own
key. The S3PathManager generates all paths for a given client and ensures that
no path ever accidentally points to another client's data.

This is the security backbone: if every path contains the client_id, and we
validate that before every read/write, data can NEVER leak between clients.

Structure:
```
s3://bucket/{client_id}/
├── raw/{upload_date}/data.csv
├── configs/mapping.yaml
├── processed/{run_date}/features.csv
├── models/{model_version}/model.tar.gz
├── outputs/scores/{run_date}/predictions.csv
└── monitoring/drift_reports/{run_date}/report.json
```
"""


class S3PathManager:
    """
    Generates and validates all S3 paths for a given client.

    Every method returns a path that contains the client_id — guaranteed.
    The validate method checks that a given path belongs to the correct
    client, preventing cross-client access.

    Think of it as the key card system: it knows which floors you're allowed
    on and rejects any attempt to visit someone else's.

    Attributes:
        bucket: The S3 bucket name.
        client_id: The client identifier (used as the top-level prefix).
        base_prefix: The full s3:// base path for this client.
    """

    def __init__(self, bucket: str, client_id: str):
        """
        Initialize the path manager for a specific client.

        Args:
            bucket: S3 bucket name (without s3:// prefix).
            client_id: Unique client identifier. All paths will be scoped
                to this client's prefix.
        """
        self.bucket = bucket
        self.client_id = client_id
        self.base_prefix = f"s3://{bucket}/{client_id}"

    def raw_data_path(self, upload_date: str) -> str:
        """
        Path where raw client CSV uploads land.

        Args:
            upload_date: Date string (e.g., "2024-01-15") identifying the upload.

        Returns:
            Full S3 path: s3://bucket/client_id/raw/{upload_date}/data.csv
        """
        return f"{self.base_prefix}/raw/{upload_date}/data.csv"

    def config_path(self) -> str:
        """
        Path to the client's approved mapping config.

        Returns:
            Full S3 path: s3://bucket/client_id/configs/mapping.yaml
        """
        return f"{self.base_prefix}/configs/mapping.yaml"

    def draft_config_path(self) -> str:
        """
        Path to the LLM-generated draft mapping (awaiting human approval).

        Returns:
            Full S3 path: s3://bucket/client_id/configs/mapping.draft.yaml
        """
        return f"{self.base_prefix}/configs/mapping.draft.yaml"

    def processed_path(self, run_date: str) -> str:
        """
        Path to processed/feature-engineered data.

        Args:
            run_date: Pipeline run date (e.g., "2024-01-15").

        Returns:
            Full S3 path: s3://bucket/client_id/processed/{run_date}/features.csv
        """
        return f"{self.base_prefix}/processed/{run_date}/features.csv"

    def model_path(self, model_version: str) -> str:
        """
        Path to a trained model artifact.

        Args:
            model_version: Version identifier (e.g., "v3" or "20240115-abc123").

        Returns:
            Full S3 path: s3://bucket/client_id/models/{model_version}/model.tar.gz
        """
        return f"{self.base_prefix}/models/{model_version}/model.tar.gz"

    def model_card_path(self, model_version: str) -> str:
        """
        Path to the model card (metadata/metrics JSON).

        Args:
            model_version: Version identifier.

        Returns:
            Full S3 path: s3://bucket/client_id/models/{model_version}/model_card.json
        """
        return f"{self.base_prefix}/models/{model_version}/model_card.json"

    def feature_artifacts_path(self, model_version: str) -> str:
        """
        Path to the serialized feature engineering artifacts.

        Args:
            model_version: Version identifier.

        Returns:
            Full S3 path: s3://bucket/client_id/models/{model_version}/feature_artifacts.pkl
        """
        return f"{self.base_prefix}/models/{model_version}/feature_artifacts.pkl"

    def output_path(self, run_date: str) -> str:
        """
        Path to the client deliverable (scored predictions CSV).

        Args:
            run_date: Pipeline run date.

        Returns:
            Full S3 path: s3://bucket/client_id/outputs/scores/{run_date}/predictions.csv
        """
        return f"{self.base_prefix}/outputs/scores/{run_date}/predictions.csv"

    def monitoring_path(self, run_date: str) -> str:
        """
        Path to drift monitoring report.

        Args:
            run_date: Pipeline run date.

        Returns:
            Full S3 path: s3://bucket/client_id/monitoring/drift_reports/{run_date}/report.json
        """
        return f"{self.base_prefix}/monitoring/drift_reports/{run_date}/report.json"

    def training_stats_path(self) -> str:
        """
        Path to training distribution stats (used as PSI reference).

        Returns:
            Full S3 path: s3://bucket/client_id/monitoring/training_stats.json
        """
        return f"{self.base_prefix}/monitoring/training_stats.json"

    def validate_path_belongs_to_client(self, path: str) -> bool:
        """
        Security check: does this path belong to our client?

        This is the key card validator. Before any read or write, we check
        that the path actually lives under our client's prefix. If it doesn't,
        something has gone wrong — maybe a bug passed the wrong client_id,
        or someone is trying to access data they shouldn't.

        Args:
            path: Any S3 path to validate.

        Returns:
            True if the path contains this client's prefix. False otherwise.
        """
        # Check both the full s3:// prefix and just the client_id as a path component
        expected_prefix = f"s3://{self.bucket}/{self.client_id}/"
        return path.startswith(expected_prefix)
