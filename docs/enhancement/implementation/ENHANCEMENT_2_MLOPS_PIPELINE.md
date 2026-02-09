# Enhancement 2: MLOps Pipeline & Automated Training

## 📋 Overview

**Enhancement Type:** Automation / MLOps  
**Priority:** 🔴 HIGH  
**Estimated Timeline:** 3-4 weeks  
**Complexity:** High  
**Dependencies:** Enhancement 1 (Feedback Collection System)

---

## 🎯 Why This Enhancement is Needed

### Current Problem:
Your current training workflow is **completely manual**:
- ❌ Must run Jupyter notebooks manually
- ❌ No automated data preprocessing
- ❌ No automated model training
- ❌ No experiment tracking
- ❌ No model versioning
- ❌ Can't trigger retraining when new data arrives
- ❌ No comparison between model versions

### Real-World Scenario Without MLOps:
```
Day 1: Users report 20 wrong predictions → You note them down
Day 7: You have 100 wrong predictions → Still noting them
Day 14: You have 500 wrong predictions → Time to retrain!

Manual Process:
1. Download images from production ⏱️ 30 minutes
2. Label them manually ⏱️ 2-3 hours
3. Run preprocessing notebook ⏱️ 1 hour
4. Merge with existing data ⏱️ 30 minutes
5. Run training notebook ⏱️ 2-4 hours
6. Evaluate results ⏱️ 30 minutes
7. If good, copy model to production ⏱️ 15 minutes
8. Restart API ⏱️ 5 minutes
9. Test in production ⏱️ 30 minutes

Total Time: 8-12 hours of manual work!
```

### With MLOps Pipeline:
```
Day 1: Users report 20 wrong predictions → Automatically saved
Day 7: 100 wrong predictions → Automatically queued
Day 14: 500 wrong predictions → Pipeline triggered automatically!

Automated Process:
1. System detects 500+ new feedback samples
2. Preprocessing runs automatically
3. Training runs automatically
4. Model evaluated automatically
5. If better: Deployed to staging automatically
6. Tests run automatically
7. If tests pass: Promoted to production
8. Notification sent to you

Total Time: 0 minutes of your time!
```

### Benefits After Implementation:
1. ✅ **Automatic retraining** - No manual intervention needed
2. ✅ **Experiment tracking** - Every training run is logged
3. ✅ **Model versioning** - Easy to compare and rollback
4. ✅ **Reproducibility** - Can reproduce any model version
5. ✅ **Faster iteration** - From weeks to days
6. ✅ **Better models** - More frequent retraining = better accuracy
7. ✅ **Data versioning** - Track which data was used for each model

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Sources                                  │
│                                                                  │
│  • User Feedback (PostgreSQL)                                   │
│  • Production Images (MinIO)                                    │
│  • Manual Uploads                                               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Trigger: New data threshold reached
                 │ Schedule: Weekly on Sunday
                 │ Manual: API call or UI button
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Apache Airflow (Workflow Orchestration)             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DAG: model_retraining_pipeline                          │  │
│  │                                                           │  │
│  │  Task 1: Check New Data ────────────────────────────┐   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 2: Data Validation & Quality Check            │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 3: Preprocessing (Tiling, Augmentation)       │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 4: Merge with Existing Dataset                │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 5: Train Model (YOLO12)                       │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 6: Evaluate on Validation Set                 │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 7: Compare with Current Model ───────────┐   │   │  │
│  │     │                                           │   │   │  │
│  │     ▼                                           │   │   │  │
│  │  Task 8: Register Model (MLflow) ◄──────── Better?  │   │  │
│  │     │                                           │   │   │  │
│  │     ▼                                           │   │   │  │
│  │  Task 9: Deploy to Staging ◄────────────────────┘   │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 10: Run Integration Tests                     │   │  │
│  │     │                                                │   │  │
│  │     ▼                                                │   │  │
│  │  Task 11: Deploy to Production ◄─── Tests Pass? ────┘  │  │
│  │     │                                                    │  │
│  │     ▼                                                    │  │
│  │  Task 12: Send Notification                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────┬──────────────────────┬───┘
                 │                    │                      │
                 ▼                    ▼                      ▼
┌─────────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│  MLflow Tracking    │  │  DVC             │  │  Model Registry    │
│                     │  │  (Data Version   │  │  (MLflow)          │
│  • Experiments      │  │   Control)       │  │                    │
│  • Metrics (mAP50)  │  │                  │  │  • Staging         │
│  • Parameters       │  │  • Dataset v1.0  │  │  • Production      │
│  • Artifacts        │  │  • Dataset v1.1  │  │  • Archived        │
│  • Visualizations   │  │  • Dataset v2.0  │  │                    │
└─────────────────────┘  └──────────────────┘  └────────────────────┘
```

---

## 📊 Technology Stack

### Core Components:

1. **Apache Airflow** - Workflow orchestration
   - Schedules and runs pipeline tasks
   - Manages dependencies between tasks
   - Provides UI for monitoring

2. **MLflow** - Experiment tracking & model registry
   - Logs all training experiments
   - Tracks metrics, parameters, artifacts
   - Manages model versions
   - Provides model registry (Staging/Production)

3. **DVC (Data Version Control)** - Data versioning
   - Version control for datasets
   - Tracks data lineage
   - Enables reproducibility

4. **Docker** - Containerization
   - Consistent environments
   - Easy deployment
   - Isolated dependencies

---

## 🔧 Implementation Guide

### Phase 1: Setup MLflow (Days 1-3)

#### Step 1.1: Install MLflow

```bash
# Install MLflow
pip install mlflow==2.10.0

# Install additional dependencies
pip install boto3 psycopg2-binary
```

Add to `requirements.txt`:
```
mlflow==2.10.0
boto3==1.34.0
psycopg2-binary==2.9.9
```

#### Step 1.2: Setup MLflow Server

**Option A: Using Docker (Recommended)**

Create `docker-compose.mlflow.yml`:

```yaml
version: '3.8'

services:
  mlflow-db:
    image: postgres:16-alpine
    container_name: mlflow_postgres
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow_user
      POSTGRES_PASSWORD: mlflow_password
    ports:
      - "5433:5432"  # Different port to avoid conflict
    volumes:
      - mlflow_db_data:/var/lib/postgresql/data

  mlflow-artifacts:
    image: minio/minio:latest
    container_name: mlflow_minio
    ports:
      - "9002:9000"  # API
      - "9003:9001"  # Console
    environment:
      MINIO_ROOT_USER: mlflow
      MINIO_ROOT_PASSWORD: mlflowpassword
    volumes:
      - mlflow_artifacts_data:/data
    command: server /data --console-address ":9001"

  mlflow-server:
    image: python:3.12-slim
    container_name: mlflow_server
    depends_on:
      - mlflow-db
      - mlflow-artifacts
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://mlflow-artifacts:9000
      - AWS_ACCESS_KEY_ID=mlflow
      - AWS_SECRET_ACCESS_KEY=mlflowpassword
    volumes:
      - ./mlflow_setup.sh:/app/setup.sh
    command: >
      bash -c "
        pip install mlflow boto3 psycopg2-binary &&
        mlflow server 
        --backend-store-uri postgresql://mlflow_user:mlflow_password@mlflow-db:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
        --host 0.0.0.0
        --port 5000
      "

volumes:
  mlflow_db_data:
  mlflow_artifacts_data:
```

Start MLflow:
```bash
docker-compose -f docker-compose.mlflow.yml up -d
```

Access MLflow UI at: `http://localhost:5000`

#### Step 1.3: Create MLflow Integration Module

Create `mlops/mlflow_integration.py`:

```python
# mlops/mlflow_integration.py

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import json

class MLflowManager:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "banana_sigatoka_detection"
    ):
        """
        Initialize MLflow tracking
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Experiment name
        """
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start MLflow run
        
        Args:
            run_name: Name for this run
            tags: Tags to add to run
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.run
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model_path: Path,
        model_name: str = "banana_sigatoka_detector",
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow
        
        Args:
            model_path: Path to model weights
            model_name: Artifact name
            registered_model_name: Name for model registry
        """
        # Log model as artifact
        mlflow.log_artifact(str(model_path), artifact_path=model_name)
        
        # If registered_model_name provided, register in model registry
        if registered_model_name:
            mlflow.register_model(
                f"runs:/{self.run.info.run_id}/{model_name}/{model_path.name}",
                registered_model_name
            )
    
    def log_artifacts(self, artifacts_dir: Path):
        """Log entire directory of artifacts"""
        mlflow.log_artifacts(str(artifacts_dir))
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        # Save as JSON artifact
        dataset_path = Path("/tmp/dataset_info.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        mlflow.log_artifact(str(dataset_path))
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
    
    def get_best_model(
        self,
        registered_model_name: str,
        metric_name: str = "val_map50",
        stage: str = "Production"
    ):
        """
        Get best model from registry
        
        Args:
            registered_model_name: Model name in registry
            metric_name: Metric to optimize
            stage: Model stage (Staging/Production/Archived)
            
        Returns:
            Model version info
        """
        client = mlflow.tracking.MlflowClient()
        
        # Get latest versions in stage
        versions = client.get_latest_versions(
            registered_model_name,
            stages=[stage]
        )
        
        if not versions:
            return None
        
        # Return latest version
        return versions[0]
    
    def promote_model(
        self,
        registered_model_name: str,
        version: int,
        stage: str = "Production"
    ):
        """
        Promote model to a stage
        
        Args:
            registered_model_name: Model name in registry
            version: Model version number
            stage: Target stage (Staging/Production)
        """
        client = mlflow.tracking.MlflowClient()
        
        # Archive current production model
        if stage == "Production":
            current_prod = client.get_latest_versions(
                registered_model_name,
                stages=["Production"]
            )
            for model in current_prod:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=model.version,
                    stage="Archived"
                )
        
        # Promote new model
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage
        )
```

#### Step 1.4: Update Training Script with MLflow

Create `training/train_with_mlflow.py`:

```python
# training/train_with_mlflow.py

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import yaml
from mlops.mlflow_integration import MLflowManager
import json

def train_model_with_tracking(
    data_yaml: str,
    base_model: str = "yolo12n.pt",
    epochs: int = 50,
    batch_size: int = 32,
    img_size: int = 736,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    experiment_name: str = "banana_sigatoka_training",
    run_name: str = None
):
    """
    Train YOLO model with MLflow tracking
    
    Args:
        data_yaml: Path to dataset YAML
        base_model: Base model weights
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Image size
        lr0: Initial learning rate
        optimizer: Optimizer name
        experiment_name: MLflow experiment name
        run_name: MLflow run name
    """
    # Initialize MLflow
    mlflow_manager = MLflowManager(experiment_name=experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run
    mlflow_manager.start_run(
        run_name=run_name,
        tags={
            "model_type": "yolo12n",
            "task": "object_detection",
            "dataset": Path(data_yaml).stem
        }
    )
    
    try:
        # Log hyperparameters
        params = {
            "model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "lr0": lr0,
            "optimizer": optimizer,
            "patience": 20,
            "warmup_epochs": 5
        }
        mlflow_manager.log_parameters(params)
        
        # Log dataset information
        with open(data_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        dataset_info = {
            "dataset_path": str(Path(data_yaml).parent),
            "num_classes": dataset_config.get('nc', 0),
            "class_names": dataset_config.get('names', []),
            "train_images": len(list(Path(dataset_config['train']).glob('*.jpg'))),
            "val_images": len(list(Path(dataset_config['val']).glob('*.jpg'))),
            "test_images": len(list(Path(dataset_config.get('test', dataset_config['val'])).glob('*.jpg')))
        }
        mlflow_manager.log_dataset_info(dataset_info)
        
        # Initialize YOLO model
        model = YOLO(base_model)
        
        # Train model
        print(f"Starting training with run name: {run_name}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=lr0,
            optimizer=optimizer,
            patience=20,
            save=True,
            plots=True,
            verbose=True,
            # Callbacks for MLflow logging
            project=f"runs/detect",
            name=run_name
        )
        
        # Log final metrics
        final_metrics = {
            "final_map50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
            "final_map50_95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            "final_precision": float(results.results_dict.get('metrics/precision(B)', 0)),
            "final_recall": float(results.results_dict.get('metrics/recall(B)', 0)),
            "final_train_loss": float(results.results_dict.get('train/box_loss', 0)),
            "final_val_loss": float(results.results_dict.get('val/box_loss', 0))
        }
        mlflow_manager.log_metrics(final_metrics)
        
        # Log training artifacts
        train_dir = Path(f"runs/detect/{run_name}")
        
        # Log model
        best_model_path = train_dir / "weights" / "best.pt"
        if best_model_path.exists():
            mlflow_manager.log_model(
                best_model_path,
                model_name="model",
                registered_model_name="banana_sigatoka_detector"
            )
        
        # Log training plots
        plots_to_log = [
            "results.png",
            "confusion_matrix.png",
            "F1_curve.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png"
        ]
        
        for plot in plots_to_log:
            plot_path = train_dir / plot
            if plot_path.exists():
                mlflow_manager.log_artifacts(train_dir)
                break
        
        print(f"Training completed. MLflow run: {mlflow_manager.run.info.run_id}")
        print(f"Results: {final_metrics}")
        
        return results, mlflow_manager.run.info.run_id
    
    finally:
        # End MLflow run
        mlflow_manager.end_run()

# Example usage
if __name__ == "__main__":
    train_model_with_tracking(
        data_yaml="combined_yolo_dataset/data.yaml",
        base_model="yolo12n.pt",
        epochs=50,
        batch_size=32,
        img_size=736,
        run_name="initial_training_v1"
    )
```

---

### Phase 2: Setup Apache Airflow (Days 4-7)

#### Step 2.1: Install Airflow

```bash
# Install Airflow
pip install apache-airflow==2.8.0
pip install apache-airflow-providers-postgres

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123
```

**Or use Docker (Recommended):**

Create `docker-compose.airflow.yml`:

```yaml
version: '3.8'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.8.0-python3.12
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./mlops:/opt/airflow/mlops
    - ./training:/opt/airflow/training
    - ./data:/opt/airflow/data
  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    &airflow-common-depends-on
    airflow-postgres:
      condition: service_healthy

services:
  airflow-postgres:
    image: postgres:16-alpine
    container_name: airflow_postgres
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    ports:
      - "5434:5432"

  airflow-webserver:
    <<: *airflow-common
    container_name: airflow_webserver
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-scheduler:
    <<: *airflow-common
    container_name: airflow_scheduler
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always
    depends_on:
      <<: *airflow-common-depends-on
      airflow-init:
        condition: service_completed_successfully

  airflow-init:
    <<: *airflow-common
    container_name: airflow_init
    entrypoint: /bin/bash
    command:
      - -c
      - |
        mkdir -p /sources/logs /sources/dags /sources/plugins
        chown -R "${AIRFLOW_UID}:0" /sources/{logs,dags,plugins}
        exec /entrypoint airflow version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_UPGRADE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: admin
      _AIRFLOW_WWW_USER_PASSWORD: admin123
    user: "0:0"
    volumes:
      - .:/sources

volumes:
  airflow-db-volume:
```

Start Airflow:
```bash
# Create directories
mkdir -p ./dags ./logs ./plugins

# Set Airflow UID
echo "AIRFLOW_UID=$(id -u)" > .env

# Start Airflow
docker-compose -f docker-compose.airflow.yml up -d
```

Access Airflow UI at: `http://localhost:8080`  
(Username: admin, Password: admin123)

#### Step 2.2: Create Training Pipeline DAG

Create `dags/model_retraining_dag.py`:

```python
# dags/model_retraining_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow')

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'banana_sigatoka_model_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline for Black Sigatoka detection',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday at midnight
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'retraining', 'yolo']
)

# Task functions

def check_new_data(**context):
    """Check if we have enough new feedback data to trigger retraining"""
    from database.connection import get_db
    from database.models import Feedback
    from datetime import datetime, timedelta
    
    # Minimum samples needed to retrain
    MIN_NEW_SAMPLES = 500
    
    # Get new feedback from last retraining
    with get_db() as db:
        # Get last retraining date (simplified - should check from MLflow)
        last_retrain = datetime.now() - timedelta(days=7)
        
        new_feedback = db.query(Feedback).filter(
            Feedback.timestamp > last_retrain,
            Feedback.processed_for_training == False
        ).count()
        
        print(f"Found {new_feedback} new feedback samples")
        
        if new_feedback >= MIN_NEW_SAMPLES:
            print(f"✅ Sufficient data ({new_feedback} >= {MIN_NEW_SAMPLES}). Proceeding with retraining.")
            return 'validate_new_data'
        else:
            print(f"❌ Insufficient data ({new_feedback} < {MIN_NEW_SAMPLES}). Skipping retraining.")
            return 'skip_retraining'

def validate_new_data(**context):
    """Validate quality of new feedback data"""
    from database.connection import get_db
    from database.models import Feedback, Prediction
    from services.storage_service import StorageService
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image
    
    storage = StorageService()
    
    with get_db() as db:
        # Get unprocessed feedback
        feedback_list = db.query(Feedback).filter(
            Feedback.processed_for_training == False,
            Feedback.is_correct == False  # Only wrong predictions
        ).all()
        
        valid_count = 0
        invalid_count = 0
        
        for feedback in feedback_list:
            # Get prediction
            prediction = db.query(Prediction).filter(
                Prediction.id == feedback.prediction_id
            ).first()
            
            if not prediction:
                invalid_count += 1
                continue
            
            # Get image from storage
            try:
                image_data = storage.get_image(prediction.image_path)
                img = Image.open(BytesIO(image_data))
                
                # Quality checks
                width, height = img.size
                if width < 256 or height < 256:
                    print(f"Image too small: {width}x{height}")
                    invalid_count += 1
                    continue
                
                # Convert to OpenCV format for blur check
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if laplacian_var < 100.0:
                    print(f"Image too blurry: {laplacian_var}")
                    invalid_count += 1
                    continue
                
                valid_count += 1
                
            except Exception as e:
                print(f"Error processing image: {e}")
                invalid_count += 1
        
        quality_rate = valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0
        
        print(f"Data Quality: {valid_count} valid, {invalid_count} invalid ({quality_rate:.2%})")
        
        # Store in XCom for next tasks
        context['ti'].xcom_push(key='valid_samples', value=valid_count)
        context['ti'].xcom_push(key='quality_rate', value=quality_rate)
        
        if quality_rate < 0.7:
            raise ValueError(f"Data quality too low: {quality_rate:.2%} < 70%")

def preprocess_new_data(**context):
    """Preprocess new feedback data"""
    from database.connection import get_db
    from database.models import Feedback, Prediction, TrainingData
    from services.storage_service import StorageService
    import shutil
    from pathlib import Path
    
    storage = StorageService()
    
    # Create temp directory for new data
    new_data_dir = Path("/tmp/new_training_data")
    new_data_dir.mkdir(exist_ok=True)
    
    # Create class directories
    class_names = ['Healthy', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6']
    for class_name in class_names:
        (new_data_dir / class_name).mkdir(exist_ok=True)
    
    with get_db() as db:
        # Get unprocessed feedback with wrong predictions
        feedback_list = db.query(Feedback).filter(
            Feedback.processed_for_training == False,
            Feedback.is_correct == False
        ).all()
        
        processed_count = 0
        
        for feedback in feedback_list:
            # Get prediction
            prediction = db.query(Prediction).filter(
                Prediction.id == feedback.prediction_id
            ).first()
            
            if not prediction or not feedback.correct_class_name:
                continue
            
            try:
                # Get image from storage
                image_data = storage.get_image(prediction.image_path)
                
                # Save to appropriate class folder
                class_dir = new_data_dir / feedback.correct_class_name
                image_filename = f"{prediction.id}_{prediction.image_hash[:8]}.jpg"
                image_path = class_dir / image_filename
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                # Mark as processed
                feedback.processed_for_training = True
                feedback.processed_at = datetime.now()
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing feedback {feedback.id}: {e}")
        
        db.commit()
        
        print(f"Preprocessed {processed_count} new images")
        context['ti'].xcom_push(key='preprocessed_count', value=processed_count)
        context['ti'].xcom_push(key='new_data_path', value=str(new_data_dir))

def merge_datasets(**context):
    """Merge new data with existing dataset"""
    from pathlib import Path
    import shutil
    import yaml
    
    # Get new data path from previous task
    new_data_path = Path(context['ti'].xcom_pull(key='new_data_path', task_ids='preprocess_new_data'))
    
    # Path to existing dataset
    existing_dataset = Path("/opt/airflow/data/combined_yolo_dataset")
    merged_dataset = Path("/tmp/merged_dataset")
    
    # Copy existing dataset
    if merged_dataset.exists():
        shutil.rmtree(merged_dataset)
    shutil.copytree(existing_dataset, merged_dataset)
    
    # Merge new data into train folder
    train_dir = merged_dataset / "train" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy new images
    new_image_count = 0
    for class_dir in new_data_path.iterdir():
        if class_dir.is_dir():
            for image_file in class_dir.glob("*.jpg"):
                shutil.copy(image_file, train_dir / image_file.name)
                new_image_count += 1
    
    print(f"Merged {new_image_count} new images into training set")
    
    # Update data.yaml
    data_yaml_path = merged_dataset / "data.yaml"
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths
    data_config['train'] = str(merged_dataset / "train" / "images")
    data_config['val'] = str(merged_dataset / "val" / "images")
    data_config['test'] = str(merged_dataset / "test" / "images")
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    context['ti'].xcom_push(key='merged_dataset_path', value=str(merged_dataset))
    context['ti'].xcom_push(key='new_image_count', value=new_image_count)

def train_model(**context):
    """Train YOLO model with new dataset"""
    from training.train_with_mlflow import train_model_with_tracking
    from datetime import datetime
    
    # Get merged dataset path
    dataset_path = context['ti'].xcom_pull(key='merged_dataset_path', task_ids='merge_datasets')
    data_yaml = f"{dataset_path}/data.yaml"
    
    # Generate run name
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Train model
    results, run_id = train_model_with_tracking(
        data_yaml=data_yaml,
        base_model="yolo12n.pt",
        epochs=50,
        batch_size=32,
        img_size=736,
        run_name=run_name
    )
    
    # Extract metrics
    final_map50 = float(results.results_dict.get('metrics/mAP50(B)', 0))
    final_map50_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0))
    
    print(f"Training completed. mAP50: {final_map50:.4f}, mAP50-95: {final_map50_95:.4f}")
    
    # Store in XCom
    context['ti'].xcom_push(key='mlflow_run_id', value=run_id)
    context['ti'].xcom_push(key='final_map50', value=final_map50)
    context['ti'].xcom_push(key='final_map50_95', value=final_map50_95)

def compare_models(**context):
    """Compare new model with current production model"""
    from mlops.mlflow_integration import MLflowManager
    
    mlflow_manager = MLflowManager()
    
    # Get new model metrics
    new_map50 = context['ti'].xcom_pull(key='final_map50', task_ids='train_model')
    
    # Get current production model
    current_model = mlflow_manager.get_best_model(
        registered_model_name="banana_sigatoka_detector",
        stage="Production"
    )
    
    if current_model is None:
        print("No production model found. Deploying new model.")
        return 'register_model'
    
    # Get current model metrics from MLflow
    import mlflow
    current_run = mlflow.get_run(current_model.run_id)
    current_map50 = current_run.data.metrics.get('final_map50', 0)
    
    print(f"Current production mAP50: {current_map50:.4f}")
    print(f"New model mAP50: {new_map50:.4f}")
    
    # Compare (with threshold to prevent frequent changes)
    IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement required
    
    if new_map50 > current_map50 + IMPROVEMENT_THRESHOLD:
        print(f"✅ New model is better by {(new_map50 - current_map50):.4f}. Proceeding with deployment.")
        return 'register_model'
    else:
        print(f"❌ New model is not significantly better. Keeping current model.")
        return 'skip_deployment'

def register_model(**context):
    """Register model in MLflow registry"""
    from mlops.mlflow_integration import MLflowManager
    import mlflow
    
    mlflow_manager = MLflowManager()
    
    # Get run ID
    run_id = context['ti'].xcom_pull(key='mlflow_run_id', task_ids='train_model')
    
    # Get model URI
    model_uri = f"runs:/{run_id}/model/best.pt"
    
    # Register model
    model_details = mlflow.register_model(
        model_uri,
        "banana_sigatoka_detector"
    )
    
    version = model_details.version
    
    # Transition to Staging
    mlflow_manager.promote_model(
        registered_model_name="banana_sigatoka_detector",
        version=version,
        stage="Staging"
    )
    
    print(f"Model registered as version {version} and promoted to Staging")
    context['ti'].xcom_push(key='model_version', value=version)

# Define tasks

check_data = BranchPythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data,
    provide_context=True,
    dag=dag
)

skip_retraining = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "Skipping retraining - insufficient data"',
    dag=dag
)

validate_data = PythonOperator(
    task_id='validate_new_data',
    python_callable=validate_new_data,
    provide_context=True,
    dag=dag
)

preprocess = PythonOperator(
    task_id='preprocess_new_data',
    python_callable=preprocess_new_data,
    provide_context=True,
    dag=dag
)

merge = PythonOperator(
    task_id='merge_datasets',
    python_callable=merge_datasets,
    provide_context=True,
    dag=dag
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

compare = BranchPythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    provide_context=True,
    dag=dag
)

skip_deployment = BashOperator(
    task_id='skip_deployment',
    bash_command='echo "Skipping deployment - model not better"',
    dag=dag
)

register = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    provide_context=True,
    dag=dag
)

# Define task dependencies
check_data >> [validate_data, skip_retraining]
validate_data >> preprocess >> merge >> train >> compare
compare >> [register, skip_deployment]
```

---

## 📊 Phase-by-Phase Implementation Schedule

### Phase 1: MLflow Setup (Week 1)

**Days 1-2:**
- ✅ Install MLflow
- ✅ Set up MLflow server (Docker)
- ✅ Configure PostgreSQL backend
- ✅ Configure MinIO for artifacts
- ✅ Test MLflow UI

**Day 3:**
- ✅ Create MLflowManager class
- ✅ Update training script with MLflow
- ✅ Test experiment tracking
- ✅ Verify artifacts are logged

### Phase 2: Airflow Setup (Week 2)

**Days 4-5:**
- ✅ Install Airflow (Docker)
- ✅ Configure Airflow database
- ✅ Create admin user
- ✅ Test Airflow UI
- ✅ Create DAGs directory structure

**Days 6-7:**
- ✅ Create retraining DAG
- ✅ Implement task functions
- ✅ Test individual tasks
- ✅ Test complete pipeline

### Phase 3: Integration & Testing (Week 3)

**Days 8-9:**
- ✅ Integrate with feedback system
- ✅ Test data flow: feedback → preprocessing → training
- ✅ Add error handling
- ✅ Add logging and monitoring

**Day 10:**
- ✅ End-to-end testing
- ✅ Performance optimization
- ✅ Documentation

### Phase 4: Advanced Features (Week 4)

**Days 11-12:**
- ✅ Add data versioning with DVC
- ✅ Implement fine-tuning capability
- ✅ Add A/B testing logic
- ✅ Implement model rollback

**Days 13-14:**
- ✅ Set up monitoring dashboard
- ✅ Add alerting for pipeline failures
- ✅ Final testing and bug fixes
- ✅ Production deployment

---

## ✅ Verification Checklist

### MLflow
- [ ] MLflow server is accessible at http://localhost:5000
- [ ] Can create experiments
- [ ] Can log parameters, metrics, artifacts
- [ ] Models are registered in registry
- [ ] Can promote models between stages

### Airflow
- [ ] Airflow UI is accessible at http://localhost:8080
- [ ] DAG appears in UI
- [ ] Can trigger DAG manually
- [ ] All tasks execute successfully
- [ ] XCom data passes between tasks

### Pipeline
- [ ] Pipeline checks for new data
- [ ] Data validation works
- [ ] Preprocessing creates valid dataset
- [ ] Training completes successfully
- [ ] Model comparison logic works
- [ ] Better models are promoted
- [ ] Worse models are rejected

---

## 📈 Success Metrics

- **Training Time**: < 4 hours for full pipeline
- **Pipeline Success Rate**: > 95% successful runs
- **Model Improvement**: New model mAP50 > old model + 2%
- **Automation Level**: 100% automated (no manual intervention)
- **Deployment Frequency**: Weekly or when 500+ new samples

---

## 🎓 Next Steps

After implementing this enhancement:

1. **Enable scheduled retraining**: Set up weekly automatic retraining
2. **Add monitoring**: Track pipeline performance
3. **Implement Enhancement 3**: CI/CD for automated deployment
4. **Scale up**: Use larger models or more data
5. **Optimize**: Fine-tune hyperparameters automatically

---

**With MLOps, your model will continuously improve without manual work!**
