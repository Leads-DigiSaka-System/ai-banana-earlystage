# mlops/mlflow_integration.py
# MLflow tracking and model registry (Enhancement 2).

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


class MLflowManager:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "banana_sigatoka_detection",
    ):
        """
        Initialize MLflow tracking.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Experiment name
        """
        mlflow.set_tracking_uri(tracking_uri)

        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Start MLflow run."""
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.run

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model_path: Path,
        model_name: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log model to MLflow.

        Args:
            model_path: Path to model weights (e.g. best.pt)
            model_name: Artifact path name
            registered_model_name: Name for model registry
        """
        mlflow.log_artifact(str(model_path), artifact_path=model_name)
        if registered_model_name:
            mlflow.register_model(
                f"runs:/{self.run.info.run_id}/{model_name}/{model_path.name}",
                registered_model_name,
            )

    def log_artifacts(self, artifacts_dir: Path) -> None:
        """Log entire directory of artifacts."""
        mlflow.log_artifacts(str(artifacts_dir))

    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset information as JSON artifact."""
        dataset_path = Path("/tmp/dataset_info.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        mlflow.log_artifact(str(dataset_path))

    def end_run(self) -> None:
        """End MLflow run."""
        mlflow.end_run()

    def get_best_model(
        self,
        registered_model_name: str,
        metric_name: str = "val_map50",
        stage: str = "Production",
    ):
        """
        Get latest model version in the given stage.

        Returns:
            Model version info or None if none in stage.
        """
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            registered_model_name,
            stages=[stage],
        )
        if not versions:
            return None
        return versions[0]

    def promote_model(
        self,
        registered_model_name: str,
        version: int,
        stage: str = "Production",
    ) -> None:
        """Promote a model version to a stage (e.g. Staging or Production)."""
        client = mlflow.tracking.MlflowClient()
        if stage == "Production":
            current = client.get_latest_versions(
                registered_model_name,
                stages=["Production"],
            )
            for model in current:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=model.version,
                    stage="Archived",
                )
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage,
        )
