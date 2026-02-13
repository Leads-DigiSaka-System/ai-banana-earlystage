# dags/model_retraining_dag.py
# Phase 2 + Phase 3: Automated model retraining pipeline (ENHANCEMENT_2_MLOPS_PIPELINE).
# Requires: Airflow containers with mlops, training, database, services mounted (see docker-airflow-official.yml).
# Phase 3: POSTGRES_HOST, MLFLOW_TRACKING_URI, STORAGE_ENDPOINT set in compose (host.docker.internal).
# Expects /opt/airflow/data/combined_yolo_dataset (YOLO dataset with data.yaml) for merge step.

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator # type: ignore[reportMissingImports]
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator  # type: ignore[reportMissingImports]
from airflow.sdk import DAG  # type: ignore[reportMissingImports]

# So DAG can import database, mlops, training, services when run inside container
sys.path.insert(0, "/opt/airflow")

log = logging.getLogger(__name__)


def _on_failure_alert(context):
    """Phase 4: On task failure, post to Slack if SLACK_WEBHOOK_URL is set."""
    import json
    import os
    import urllib.request

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    try:
        ti = context.get("task_instance")
        dag = context.get("dag")
        task_id = getattr(ti, "task_id", "?") if ti else "?"
        dag_id = getattr(dag, "dag_id", "?") if dag else "?"
        exec_dt = str(context.get("execution_date", ""))
        msg = {"text": f"Airflow task failed: DAG={dag_id} task={task_id} execution_date={exec_dt}"}
        data = json.dumps(msg).encode("utf-8")
        req = urllib.request.Request(webhook, data=data, method="POST", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning("Slack alert failed: %s", e)


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,  # Set True and AIRFLOW__SMTP__* when using email
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure_alert,
}

dag = DAG(
    "banana_sigatoka_model_retraining",
    default_args=default_args,
    description="Automated model retraining pipeline for Black Sigatoka detection",
    schedule="0 0 * * 0",  # Weekly Sunday midnight (cron)
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining", "yolo"],
)


def check_new_data(**kwargs):
    """Check if we have enough new feedback data to trigger retraining."""
    from datetime import datetime, timedelta

    from database.connection import get_db
    from database.models import Feedback

    ti = kwargs["ti"]
    MIN_NEW_SAMPLES = 500
    log.info("check_new_data: connecting to feedback DB and counting new samples (last 7 days, unprocessed)")

    try:
        with get_db() as db:
            last_retrain = datetime.now() - timedelta(days=7)
            new_feedback = (
                db.query(Feedback)
                .filter(
                    Feedback.timestamp > last_retrain,
                    Feedback.processed_for_training == False,
                )
                .count()
            )
        log.info("check_new_data: found %s new feedback samples (min required=%s)", new_feedback, MIN_NEW_SAMPLES)
        print(f"Found {new_feedback} new feedback samples")
        if new_feedback >= MIN_NEW_SAMPLES:
            log.info("Proceeding to validate_new_data")
            return "validate_new_data"
        log.info("Skipping retraining (insufficient data)")
        return "skip_retraining"
    except Exception as e:
        log.exception("check_new_data failed: %s", e)
        raise


def validate_new_data(**kwargs):
    """Validate quality of new feedback data."""
    from io import BytesIO

    import cv2
    import numpy as np
    from PIL import Image

    from database.connection import get_db
    from database.models import Feedback, Prediction
    from services.storage_service import StorageService

    ti = kwargs["ti"]
    log.info("validate_new_data: starting quality checks (size, blur)")
    storage = StorageService()
    valid_count = 0
    invalid_count = 0

    try:
        with get_db() as db:
            feedback_list = (
                db.query(Feedback)
                .filter(
                    Feedback.processed_for_training == False,
                    Feedback.is_correct == False,
                )
                .all()
            )
            for feedback in feedback_list:
                prediction = (
                    db.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
                )
                if not prediction:
                    invalid_count += 1
                    continue
                try:
                    image_data = storage.get_image(prediction.image_path)
                    img = Image.open(BytesIO(image_data))
                    w, h = img.size
                    if w < 256 or h < 256:
                        invalid_count += 1
                        continue
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < 100.0:
                        invalid_count += 1
                        continue
                    valid_count += 1
                except Exception as e:
                    log.warning("Image validation failed for one sample: %s", e)
                    invalid_count += 1

        total = valid_count + invalid_count
        quality_rate = valid_count / total if total > 0 else 0
        log.info("validate_new_data: valid=%s invalid=%s quality_rate=%.2f%%", valid_count, invalid_count, quality_rate * 100)
        print(f"Data Quality: {valid_count} valid, {invalid_count} invalid ({quality_rate:.2%})")
        ti.xcom_push(key="valid_samples", value=valid_count)
        ti.xcom_push(key="quality_rate", value=quality_rate)
        if quality_rate < 0.7:
            raise ValueError(f"Data quality too low: {quality_rate:.2%} < 70%")
    except ValueError:
        raise
    except Exception as e:
        log.exception("validate_new_data failed: %s", e)
        raise


def preprocess_new_data(**kwargs):
    """Preprocess new feedback data into class folders."""
    from pathlib import Path

    from database.connection import get_db
    from database.models import Feedback, Prediction
    from services.storage_service import StorageService

    ti = kwargs["ti"]
    log.info("preprocess_new_data: exporting feedback images to /tmp/new_training_data")
    try:
        storage = StorageService()
        new_data_dir = Path("/tmp/new_training_data")
        new_data_dir.mkdir(exist_ok=True)
        class_names = ["Healthy", "Stage1", "Stage2", "Stage3", "Stage4", "Stage5", "Stage6"]
        for c in class_names:
            (new_data_dir / c).mkdir(exist_ok=True)

        processed_count = 0
        with get_db() as db:
            feedback_list = (
                db.query(Feedback)
                .filter(
                    Feedback.processed_for_training == False,
                    Feedback.is_correct == False,
                )
                .all()
            )
            for feedback in feedback_list:
                prediction = (
                    db.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
                )
                if not prediction or not feedback.correct_class_name:
                    continue
                try:
                    image_data = storage.get_image(prediction.image_path)
                    class_dir = new_data_dir / feedback.correct_class_name
                    fname = f"{prediction.id}_{prediction.image_hash[:8]}.jpg"
                    (class_dir / fname).write_bytes(image_data)
                    feedback.processed_for_training = True
                    feedback.processed_at = datetime.now()
                    processed_count += 1
                except Exception as e:
                    log.warning("preprocess feedback id=%s failed: %s", feedback.id, e)
            db.commit()

        log.info("preprocess_new_data: exported %s images", processed_count)
        print(f"Preprocessed {processed_count} new images")
        ti.xcom_push(key="preprocessed_count", value=processed_count)
        ti.xcom_push(key="new_data_path", value=str(new_data_dir))
    except Exception as e:
        log.exception("preprocess_new_data failed: %s", e)
        raise


def merge_datasets(**kwargs):
    """Merge new data with existing YOLO dataset."""
    import shutil
    from pathlib import Path

    import yaml

    ti = kwargs["ti"]
    new_data_path = Path(
        ti.xcom_pull(key="new_data_path", task_ids="preprocess_new_data")
    )
    existing_dataset = Path("/opt/airflow/data/combined_yolo_dataset")
    merged_dataset = Path("/tmp/merged_dataset")
    log.info("merge_datasets: existing=%s merged=%s", existing_dataset, merged_dataset)

    try:
        if merged_dataset.exists():
            shutil.rmtree(merged_dataset)
        if not existing_dataset.exists():
            raise FileNotFoundError(
                f"Expected existing dataset at {existing_dataset}. "
                "Create combined_yolo_dataset with data.yaml, train/val structure."
            )
        shutil.copytree(existing_dataset, merged_dataset)
        train_dir = merged_dataset / "train" / "images"
        train_dir.mkdir(parents=True, exist_ok=True)
        new_image_count = 0
        for class_dir in new_data_path.iterdir():
            if class_dir.is_dir():
                for f in class_dir.glob("*.jpg"):
                    shutil.copy(f, train_dir / f.name)
                    new_image_count += 1
        log.info("merge_datasets: merged %s new images", new_image_count)
        print(f"Merged {new_image_count} new images")
        # Ensure val/test dirs exist so data.yaml paths are valid (YOLO may not have test in base)
        (merged_dataset / "val" / "images").mkdir(parents=True, exist_ok=True)
        (merged_dataset / "test" / "images").mkdir(parents=True, exist_ok=True)
        data_yaml_path = merged_dataset / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path) as f:
                data_config = yaml.safe_load(f)
            data_config["train"] = str(merged_dataset / "train" / "images")
            data_config["val"] = str(merged_dataset / "val" / "images")
            data_config["test"] = str(merged_dataset / "test" / "images")
            with open(data_yaml_path, "w") as f:
                yaml.dump(data_config, f)
        ti.xcom_push(key="merged_dataset_path", value=str(merged_dataset))
        ti.xcom_push(key="new_image_count", value=new_image_count)
    except FileNotFoundError:
        raise
    except Exception as e:
        log.exception("merge_datasets failed: %s", e)
        raise


def pull_data_dvc(**kwargs):
    """Phase 4: Pull dataset from DVC remote when DVC_ENABLED=1."""
    import os
    import subprocess

    if os.environ.get("DVC_ENABLED", "").strip() != "1":
        log.info("pull_data_dvc: DVC_ENABLED not set, skipping dvc pull")
        return
    try:
        # Run from project root (where dvc.yaml lives); /opt/airflow is project root in container
        cwd = "/opt/airflow"
        r = subprocess.run(
            ["dvc", "pull"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            log.warning("dvc pull failed: %s %s", r.stdout, r.stderr)
            raise RuntimeError(f"dvc pull failed: {r.stderr}")
        log.info("pull_data_dvc: dvc pull succeeded")
    except FileNotFoundError:
        log.warning("pull_data_dvc: dvc not installed, skipping")
    except Exception as e:
        log.exception("pull_data_dvc failed: %s", e)
        raise


def train_model(**kwargs):
    """Train YOLO model with MLflow tracking. Phase 4: fine-tune from Production when FINETUNE_FROM_PRODUCTION=1."""
    import os
    from pathlib import Path

    from mlops.mlflow_integration import MLflowManager
    from training.train_with_mlflow import train_model_with_tracking

    ti = kwargs["ti"]
    dataset_path = ti.xcom_pull(key="merged_dataset_path", task_ids="merge_datasets")
    if not dataset_path:
        raise ValueError("merge_datasets did not push merged_dataset_path; check upstream task")
    data_yaml = f"{dataset_path}/data.yaml"
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info("train_model: starting run_name=%s data_yaml=%s", run_name, data_yaml)

    base_model = "yolo12n.pt"
    if os.environ.get("FINETUNE_FROM_PRODUCTION", "").strip() == "1":
        try:
            mgr = MLflowManager()
            weights_path = mgr.download_model_weights(
                registered_model_name="banana_sigatoka_detector",
                stage="Production",
                artifact_path="model",
                dest_dir=Path("/tmp/finetune_base"),
            )
            if weights_path and weights_path.exists():
                base_model = str(weights_path)
                log.info("train_model: fine-tuning from %s", base_model)
        except Exception as e:
            log.warning("Fine-tuning from Production failed, using yolo12n.pt: %s", e)

    try:
        results, run_id = train_model_with_tracking(
            data_yaml=data_yaml,
            base_model=base_model,
            epochs=50,
            batch_size=32,
            img_size=736,
            run_name=run_name,
        )
        rd = getattr(results, "results_dict", None) or getattr(results, "results", None) or {}
        final_map50 = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0)))
        final_map50_95 = float(
            rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", 0))
        )
        log.info("train_model: run_id=%s mAP50=%.4f mAP50-95=%.4f", run_id, final_map50, final_map50_95)
        print(f"Training completed. mAP50: {final_map50:.4f}, mAP50-95: {final_map50_95:.4f}")
        ti.xcom_push(key="mlflow_run_id", value=run_id)
        ti.xcom_push(key="final_map50", value=final_map50)
        ti.xcom_push(key="final_map50_95", value=final_map50_95)
    except Exception as e:
        log.exception("train_model failed: %s", e)
        raise


def compare_models(**kwargs):
    """Compare new model with current production; return next task id."""
    import mlflow

    from mlops.mlflow_integration import MLflowManager

    ti = kwargs["ti"]
    new_map50 = ti.xcom_pull(key="final_map50", task_ids="train_model")
    log.info("compare_models: new mAP50=%.4f", new_map50)

    try:
        mlflow_manager = MLflowManager()
        current_model = mlflow_manager.get_best_model(
            registered_model_name="banana_sigatoka_detector",
            stage="Production",
        )
        if current_model is None:
            log.info("No production model; will register new model")
            return "register_model"
        current_run = mlflow.get_run(current_model.run_id)
        current_map50 = current_run.data.metrics.get("final_map50", 0)
        log.info("compare_models: current mAP50=%.4f new mAP50=%.4f", current_map50, new_map50)
        print(f"Current production mAP50: {current_map50:.4f}, New mAP50: {new_map50:.4f}")
        IMPROVEMENT_THRESHOLD = 0.02
        if new_map50 > current_map50 + IMPROVEMENT_THRESHOLD:
            log.info("New model better; proceeding to register_model")
            return "register_model"
        log.info("New model not better; skipping deployment")
        return "skip_deployment"
    except Exception as e:
        log.exception("compare_models failed: %s", e)
        raise


def register_model(**kwargs):
    """Register model in MLflow and promote to Staging."""
    import mlflow

    from mlops.mlflow_integration import MLflowManager

    ti = kwargs["ti"]
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")
    log.info("register_model: run_id=%s", run_id)

    try:
        model_uri = f"runs:/{run_id}/model/best.pt"
        model_details = mlflow.register_model(model_uri, "banana_sigatoka_detector")
        version = model_details.version
        MLflowManager().promote_model(
            registered_model_name="banana_sigatoka_detector",
            version=version,
            stage="Staging",
        )
        log.info("register_model: version=%s promoted to Staging", version)
        print(f"Model registered as version {version}, promoted to Staging")
        ti.xcom_push(key="model_version", value=version)
    except Exception as e:
        log.exception("register_model failed: %s", e)
        raise


def run_integration_tests(**kwargs):
    """Phase 4: Smoke test API (health + optional predict). Set INTEGRATION_TEST_URL or skip."""
    import os
    import urllib.request

    ti = kwargs["ti"]
    url = os.environ.get("INTEGRATION_TEST_URL", "").strip()
    if not url:
        log.info("run_integration_tests: INTEGRATION_TEST_URL not set, skipping")
        return
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as r:
            if r.status != 200:
                raise RuntimeError(f"Health returned {r.status}")
        log.info("run_integration_tests: health check passed")
    except Exception as e:
        log.exception("run_integration_tests failed: %s", e)
        raise


def promote_to_production(**kwargs):
    """Phase 4: Promote the model we just registered (Staging) to Production."""
    from mlops.mlflow_integration import MLflowManager

    ti = kwargs["ti"]
    version = ti.xcom_pull(key="model_version", task_ids="register_model")
    if version is None:
        raise ValueError("No model_version from register_model")
    log.info("promote_to_production: promoting version=%s to Production", version)
    MLflowManager().promote_model(
        registered_model_name="banana_sigatoka_detector",
        version=int(version),
        stage="Production",
    )
    log.info("promote_to_production: done")


def deploy_to_production(**kwargs):
    """Phase 4: Copy Production model from MLflow to DEPLOY_MODEL_PATH (e.g. for API)."""
    import os
    import subprocess
    from pathlib import Path

    dest = os.environ.get("DEPLOY_MODEL_PATH", "/opt/airflow/models/weights/best.pt")
    log.info("deploy_to_production: dest=%s", dest)
    try:
        # Run deploy script so we use same MLflow env as Airflow
        subprocess.run(
            [
                "python", "-m", "scripts.deploy_production_model",
                "--dest", dest,
            ],
            cwd="/opt/airflow",
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
        log.info("deploy_to_production: succeeded")
    except subprocess.CalledProcessError as e:
        log.exception("deploy_to_production failed: %s %s", e.stdout, e.stderr)
        raise
    except Exception as e:
        log.exception("deploy_to_production failed: %s", e)
        raise


# Tasks
check_data = BranchPythonOperator(
    task_id="check_new_data",
    python_callable=check_new_data,
    dag=dag,
)
skip_retraining = BashOperator(
    task_id="skip_retraining",
    bash_command='echo "Skipping retraining - insufficient data"',
    dag=dag,
)
validate_data = PythonOperator(
    task_id="validate_new_data",
    python_callable=validate_new_data,
    dag=dag,
)
preprocess = PythonOperator(
    task_id="preprocess_new_data",
    python_callable=preprocess_new_data,
    dag=dag,
)
pull_dvc = PythonOperator(
    task_id="pull_data_dvc",
    python_callable=pull_data_dvc,
    dag=dag,
)
merge = PythonOperator(
    task_id="merge_datasets",
    python_callable=merge_datasets,
    dag=dag,
)
train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)
compare = BranchPythonOperator(
    task_id="compare_models",
    python_callable=compare_models,
    dag=dag,
)
skip_deployment = BashOperator(
    task_id="skip_deployment",
    bash_command='echo "Skipping deployment - model not better"',
    dag=dag,
)
register = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    dag=dag,
)
integration_tests = PythonOperator(
    task_id="run_integration_tests",
    python_callable=run_integration_tests,
    dag=dag,
)
promote_prod = PythonOperator(
    task_id="promote_to_production",
    python_callable=promote_to_production,
    dag=dag,
)
deploy_prod = PythonOperator(
    task_id="deploy_to_production",
    python_callable=deploy_to_production,
    dag=dag,
)

# Dependencies
check_data >> [validate_data, skip_retraining]
validate_data >> preprocess >> pull_dvc >> merge >> train >> compare
compare >> [register, skip_deployment]
register >> integration_tests >> promote_prod >> deploy_prod
