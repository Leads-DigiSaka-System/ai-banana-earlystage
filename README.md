# Black Sigatoka Early Stage Detection System

A production-ready machine learning system for detecting and classifying Black Sigatoka disease stages in banana leaves using YOLO object detection. The system identifies 7 distinct disease stages: Healthy, Stage1, Stage2, Stage3, Stage4, Stage5, and Stage6.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Model Training Workflow](#model-training-workflow)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements an end-to-end machine learning pipeline for agricultural disease detection, specifically targeting Black Sigatoka (Mycosphaerella fijiensis) in banana plants. The system combines:

- **Computer Vision**: YOLO12-based object detection model
- **REST API**: FastAPI-based inference service
- **Data Pipeline**: Automated preprocessing, augmentation, and dataset management
- **Production Deployment**: Dockerized containerization for scalable deployment

### Key Capabilities

- **Multi-stage Classification**: Detects 7 disease progression stages
- **Real-time Inference**: Fast API response times with optimized model serving
- **Image Tiling**: Handles high-resolution images through intelligent tiling
- **Feedback Collection**: Submit correct/incorrect feedback per prediction; stats and accuracy from user feedback
- **Data Storage**: PostgreSQL for predictions, feedback, training_data; optional MinIO for image storage
- **Production Ready**: Docker deployment with health checks and monitoring

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│              (Web, Mobile, IoT Devices)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP/REST API
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Application                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Router     │  │  Validation  │  │  Formatting  │    │
│  │  (Endpoints) │  │   Service    │  │   Service    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                          │                                │
│                  ┌───────▼────────┐                       │
│                  │ Detection       │                       │
│                  │ Service         │                       │
│                  └───────┬────────┘                       │
└──────────────────────────┼────────────────────────────────┘
                           │
                           │
┌──────────────────────────▼────────────────────────────────┐
│              Inference Engine                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Image Processing  │  Model Inference  │  Tiling    │  │
│  │  (Preprocessing)  │  (YOLO12)        │  (256x256)  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────┬────────────────────────────────┘
                           │
                           │
┌──────────────────────────▼────────────────────────────────┐
│              Model Storage                                │
│              models/weights/best.pt                        │
└───────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Framework**: FastAPI 0.124+
- **ML Framework**: Ultralytics YOLO (YOLO12n)
- **Deep Learning**: PyTorch 2.9+
- **Image Processing**: OpenCV 4.11+, Pillow 12.0+
- **Containerization**: Docker, Docker Compose
- **Python**: 3.12+

---

## Features

### Core Functionality

1. **Disease Stage Detection**
   - 7-class classification: Healthy, Stage1 through Stage6
   - Bounding box localization for disease regions
   - Confidence scoring for each detection

2. **Image Processing**
   - Automatic image validation (format, size, quality)
   - Intelligent tiling for high-resolution images (256x256 tiles)
   - Support for JPG, PNG, JPEG formats
   - Automatic resizing to model input size (736x736)

3. **API Endpoints**
   - Full detection with bounding boxes
   - Classification-only endpoint
   - Feedback: submit (correct/wrong + optional correct class), stats (accuracy, per-class)
   - Database CRUD: predictions, feedback, training_data, model_performance (list/get/patch/delete)
   - Health monitoring and model information
   - User tracking support (optional `user_id` returns `prediction_id` for feedback)

4. **Production Features**
   - Docker containerization
   - Health check endpoints
   - CORS support for web applications
   - Error handling and validation
   - Model caching for performance

---

## Prerequisites

### System Requirements

- **Python**: 3.12 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (optional, for easier deployment)
- **GPU**: CUDA-capable GPU recommended for training (optional for inference)

### Model Requirements

- Trained YOLO model weights (`best.pt`) in `models/weights/` directory
- Model should be trained on 736x736 input size
- Model should support 7 classes (Healthy, Stage1-6)

### Optional (Feedback & Storage)

- **PostgreSQL**: For saving predictions and feedback (see `.env.example`). Use `scripts/setup_database.py` to create DB and tables.
- **MinIO**: For object storage of prediction images. If `STORAGE_ENDPOINT` is not set, images are stored locally under `data/uploads/predictions/`.

---

## Installation

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ai-banana-earlystage

# Ensure model weights are present
ls models/weights/best.pt

# Build and run with Docker Compose
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-banana-earlystage

# Install dependencies using uv (recommended)
pip install uv
uv pip install -e .

# Or using pip
pip install -r requirements.txt

# Ensure model weights are present
ls models/weights/best.pt

# Run the application
python main.py
```

The API will be available at `http://localhost:8000`

---

## Quick Start

### 1. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","message":"API is running"}
```

### 2. Check Model Information

```bash
curl http://localhost:8000/model/info

# Returns class names and model configuration
```

### 3. Test Classification

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/predict/classify" \
  -F "file=@path/to/banana_leaf.jpg" \
  -F "user_id=test_user"

# Using Python
import requests

with open('banana_leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/classify',
        files={'file': f},
        data={'user_id': 'test_user'}
    )
    print(response.json())
```

When `user_id` is provided, the response includes `prediction_id` (for feedback).

### 4. Submit Feedback (optional, requires PostgreSQL)

```bash
# After a prediction that returned prediction_id:
curl -X POST "http://localhost:8000/api/v1/feedback/submit" \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "<prediction_id>", "is_correct": false, "correct_class_name": "Stage3"}'

# Get feedback stats
curl "http://localhost:8000/api/v1/feedback/stats?days=7"
```

### 5. Access API Documentation

Open `http://localhost:8000/docs` in your browser for interactive API documentation.

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "ok",
  "message": "API is running"
}
```

#### 2. Model Information

**GET** `/model/info`

Get model configuration and class information.

**Response:**
```json
{
  "classes": {
    "0": "Healthy",
    "1": "Stage1",
    "2": "Stage2",
    "3": "Stage3",
    "4": "Stage4",
    "5": "Stage5",
    "6": "Stage6"
  },
  "num_classes": 7,
  "class_names": ["Healthy", "Stage1", "Stage2", "Stage3", "Stage4", "Stage5", "Stage6"]
}
```

#### 3. Full Detection

**POST** `/api/v1/predict`

Detect disease stages with bounding box coordinates.

**Request:**
- `file` (multipart/form-data): Image file (JPG, PNG, JPEG)
- `user_id` (optional, form-data): User identifier for tracking

**Response:**
```json
{
  "success": true,
  "user_id": "test_user",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "class_id": 2,
      "class_name": "Stage2",
      "confidence": 0.85,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 350.7,
        "y2": 450.9
      }
    }
  ]
}
```

#### 4. Classification Only

**POST** `/api/v1/predict/classify`

Get disease classification without bounding boxes (simplified output).

**Request:**
- `file` (multipart/form-data): Image file
- `user_id` (optional, form-data): User identifier

**Response:**
```json
{
  "user_id": "test_user",
  "class_name": "Stage2",
  "confidence": 0.85
}
```

#### 5. Feedback Submit

**POST** `/api/v1/feedback/submit`

Submit user feedback for a prediction (requires DB). Use `prediction_id` from `/predict` or `/predict/classify` when `user_id` was sent.

**Request body (JSON):**
- `prediction_id` (required): UUID from prediction response
- `is_correct` (required): `true` or `false`
- `correct_class_name` (optional): e.g. `"Stage3"` when wrong
- `correct_class_id` (optional): e.g. `3`
- `user_comment` (optional): Free text
- `confidence_rating` (optional): 1–5

**Response:**
```json
{
  "success": true,
  "feedback_id": "uuid",
  "message": "Thank you for your feedback!"
}
```

#### 6. Feedback Stats

**GET** `/api/v1/feedback/stats?days=7&model_version=`

Returns accuracy and per-class statistics from feedback. Query params: `days` (default 7), `model_version` (optional).

#### 7. Database CRUD

- **GET** `/api/v1/db/predictions` — List/search predictions
- **GET** `/api/v1/db/predictions/{id}` — Get one prediction
- **PATCH** `/api/v1/db/predictions/{id}` — Update prediction
- **DELETE** `/api/v1/db/predictions/{id}` — Delete prediction

Same pattern for `/api/v1/db/training-data`, `/api/v1/db/model-performance`, `/api/v1/db/feedback`. See interactive docs at `/docs`.

#### 8. Debug Endpoint

**POST** `/api/v1/predict/classify/debug`

Debug endpoint with low confidence threshold to diagnose detection issues.

**Response:**
```json
{
  "user_id": "test_user",
  "all_detections": [...],
  "classes_detected": [...],
  "total_detections": 5,
  "top_class": "Stage2",
  "top_confidence": 0.85
}
```

### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Error message describing the issue"
}
```

Common errors:
- Invalid image format
- Image file too large
- Image dimensions exceed limits
- Model not found

---

## Model Training Workflow

The model training process consists of three main phases:

### Phase 1: Data Preprocessing

**Notebook:** `notebook/data-labeling-classification.ipynb`

**Steps:**
1. **Image Quality Assessment**: Filter images based on blur, brightness, resolution, and file size
2. **Image Tiling**: Split large images into 256x256 tiles for increased training data
3. **Data Splitting**: Stratified split into 70% train / 15% validation / 15% test
4. **Data Augmentation**: Apply rotation, brightness, crop/zoom, and horizontal flip to training set
5. **YOLO Format Conversion**: Convert annotations to YOLO format with `data.yaml` configuration

**Output:** `yolo_classification_dataset/` directory with YOLO-formatted data

### Phase 2: Dataset Merging (Optional)

**Notebook:** `notebook/bsed-datasets-merge.ipynb`

**Purpose:** Combine multiple datasets with unified class mapping

**Class Mapping:**
- Functional → Healthy (ID: 0)
- Stage1 → Stage1 (ID: 1)
- Stage2 → Stage2 (ID: 2)
- Stage3 → Stage3 (ID: 3)
- Mild → Stage4 (ID: 4)
- Moderate → Stage5 (ID: 5)
- Severe → Stage6 (ID: 6)

**Output:** `combined_yolo_dataset/` with 7 unified classes

### Phase 3: Model Training

**Notebook:** `notebook/bsed-training.ipynb`

**Configuration:**
- **Model**: YOLO12n (nano variant)
- **Input Size**: 736x736 pixels
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (initial) with cosine annealing
- **Epochs**: 10 (configurable)
- **Early Stopping**: Patience = 20 epochs

**Training Process:**
1. Load and verify dataset split proportions
2. Initialize YOLO model
3. Configure hyperparameters
4. Train with validation monitoring
5. Save best model weights
6. Evaluate on test set (once only)
7. Generate performance metrics and visualizations

**Output:**
- `best.pt`: Best model weights based on validation mAP
- `last.pt`: Final epoch weights
- Training metrics and visualizations in `runs/detect/`

### Data Split Strategy

**Proportions:** 70% Training / 15% Validation / 15% Test

**Rationale:**
- **70% Training**: Maximum data for model learning
- **15% Validation**: Sufficient for hyperparameter tuning and early stopping
- **15% Test**: Adequate for unbiased final evaluation

**Important:** The test set should only be used once for final evaluation after training is complete.

---

## Configuration

### Model Configuration (`config.py`)

```python
# Model paths and thresholds
MODEL_PATH = Path("models/weights/best.pt")
MODEL_CONFIDENCE = 0.25  # Detection confidence threshold
MODEL_IOU = 0.7  # IoU threshold for NMS
MODEL_IMAGE_SIZE = 736  # Input image size (must match training)

# Tiling configuration
USE_TILING = True  # Enable image tiling for inference
TILE_SIZE = 256  # Tile size (matches training)
TILE_OVERLAP = 0.1  # 10% overlap between tiles

# Class names
CLASS_NAMES = {
    0: "Healthy",
    1: "Stage1",
    2: "Stage2",
    3: "Stage3",
    4: "Stage4",
    5: "Stage5",
    6: "Stage6"
}

# API configuration
HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_DIMENSION = 10000  # Maximum width/height in pixels
MAX_IMAGE_MEMORY_MB = 50  # Maximum file size in MB
```

### Environment Variables

Copy `.env.example` to `.env` and fill in values. Key variables:

```env
# Server (optional)
HOST=0.0.0.0
PORT=8000

# PostgreSQL (for feedback & predictions storage)
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=ai_banana_early_stage
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# MinIO (optional; if unset, images stored under data/uploads/predictions/)
STORAGE_ENDPOINT=localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin123
STORAGE_SECURE=false
STORAGE_BUCKET=ai-banana-early-stage
```

Run `scripts/setup_database.py` to create the database and tables when using feedback features.

### Running Tests

```bash
# Install dev deps (includes pytest, httpx)
uv sync

# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test modules
uv run pytest tests/unit/test_feedback_service.py tests/unit/test_storage_service.py -v
```

---

## Deployment

### Docker Deployment

#### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

#### Using Docker Directly

```bash
# Build image
docker build -t banana-disease-api .

# Run container
docker run -d \
  --name banana-disease-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  banana-disease-api
```

### Production Considerations

1. **Security**
   - Restrict CORS origins in production
   - Implement authentication/authorization
   - Use HTTPS/TLS
   - Validate and sanitize all inputs

2. **Performance**
   - Use GPU acceleration if available
   - Implement request queuing for high traffic
   - Consider model quantization for faster inference
   - Use load balancing for multiple instances

3. **Monitoring**
   - Set up health check monitoring
   - Log API requests and errors
   - Monitor model performance metrics
   - Track inference latency

4. **Scaling**
   - Use container orchestration (Kubernetes, Docker Swarm)
   - Implement horizontal scaling
   - Use shared model storage (S3, NFS)
   - Consider model serving frameworks (TensorFlow Serving, TorchServe)

For detailed deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

---

## Project Structure

```
ai-banana-earlystage/
├── Data/                          # Raw image data
│   └── Sigatoka pics/
├── data/uploads/predictions/      # Local prediction images (when MinIO not used)
├── database/                      # DB connection & models
│   ├── connection.py             # get_db, session
│   ├── models.py                 # Prediction, Feedback, TrainingData
│   └── model_performance.py      # ModelPerformance
├── docs/                          # Documentation
│   ├── PHASE_CONNECTION_CHECKLIST.md
│   ├── ENHANCEMENT_1_IMPLEMENTATION_STATUS.md
│   └── enhancement/implementation/  # Enhancement 1–5 guides
├── notebook/                      # Jupyter notebooks for training
│   ├── data-labeling-classification.ipynb
│   ├── bsed-datasets-merge.ipynb
│   └── bsed-training.ipynb
├── router/                        # API routes
│   ├── process.py                # /api/v1/predict, /predict/classify
│   ├── feedback.py               # /api/v1/feedback/submit, /stats
│   └── database_crud.py          # /api/v1/db/* (predictions, feedback, etc.)
├── schemas/                       # Pydantic request/response
│   └── feedback.py               # FeedbackSubmitRequest
├── services/                      # Core service modules
│   ├── detection_service.py      # Main detection logic
│   ├── inference.py              # Model inference
│   ├── feedback_service.py       # save_prediction, save_feedback, get_feedback_stats
│   ├── storage_service.py        # MinIO/local image storage, read_image_bytes
│   ├── image_processing.py       # Image preprocessing
│   ├── validation.py             # Input validation
│   └── formatting.py             # Response formatting
├── scripts/                       # DB & utility scripts
│   ├── setup_database.py         # Create DB and tables
│   └── export_feedback_for_training.py  # TODO: export for retraining
├── tests/unit/                    # Unit tests
│   ├── test_feedback_service.py
│   ├── test_feedback_schemas.py
│   ├── test_storage_service.py
│   └── test_database_crud.py
├── models/weights/               # Model storage (best.pt, last.pt)
├── main.py                        # FastAPI app entry
├── config.py                      # Configuration
├── pyproject.toml                 # Project metadata & deps
├── .env.example                   # Env template (Postgres, MinIO)
├── Dockerfile
├── docker-compose.yml
├── README.md                      # This file
├── DOCKER_DEPLOYMENT.md
├── DOCKER_TROUBLESHOOTING.md
└── BUILD_LOCAL.md
```

---

## Documentation

### Additional Documentation

- **[docs/ENHANCEMENT_1_IMPLEMENTATION_STATUS.md](docs/ENHANCEMENT_1_IMPLEMENTATION_STATUS.md)**: Status of Feedback Collection implementation vs enhancement doc; router and API summary.
- **[docs/PHASE_CONNECTION_CHECKLIST.md](docs/PHASE_CONNECTION_CHECKLIST.md)**: How predict → save → feedback → storage connect; use of `read_image_bytes` for export.
- **Enhancement guides** (docs/enhancement/implementation/): ENHANCEMENT_1 (Feedback), ENHANCEMENT_2 (MLOps), ENHANCEMENT_3 (DevOps/CI-CD), ENHANCEMENT_4 (Training), ENHANCEMENT_5 (Monitoring).
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)**: Docker deployment (Dockerfile, Compose, production tips).
- **[DOCKER_TROUBLESHOOTING.md](DOCKER_TROUBLESHOOTING.md)**: Docker troubleshooting.
- **[BUILD_LOCAL.md](BUILD_LOCAL.md)**: Local build and testing instructions.

---

## Performance Metrics

### Model Performance Targets

- **mAP50**: > 0.7 (Good: >0.7, Excellent: >0.8)
- **mAP50-95**: > 0.5
- **Precision**: > 0.7
- **Recall**: > 0.7
- **F1 Score**: > 0.7

### Inference Performance

- **Latency**: < 500ms per image (CPU), < 100ms (GPU)
- **Throughput**: 10+ requests/second (CPU), 50+ requests/second (GPU)
- **Memory**: ~2GB RAM (CPU), ~4GB VRAM (GPU)

*Note: Actual performance depends on hardware, image size, and model variant.*

---

## Troubleshooting

### Common Issues

#### Model Not Found

**Error:** `Model not found at models/weights/best.pt`

**Solution:**
```bash
# Ensure model file exists
ls -lh models/weights/best.pt

# If missing, copy trained model to this location
cp path/to/trained/best.pt models/weights/best.pt
```

#### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
# Linux/Mac:
lsof -i :8000

# Windows:
netstat -ano | findstr :8000

# Change port in config.py or docker-compose.yml
```

#### Image Processing Errors

**Error:** `Could not decode image` or `Image file too large`

**Solution:**
- Verify image format (JPG, PNG, JPEG)
- Check image file size (< 50MB)
- Ensure image dimensions are reasonable (< 10000px)
- Verify image is not corrupted

#### Docker Build Fails

**Error:** Build errors during Docker image creation

**Solution:**
```bash
# Clean build without cache
docker-compose build --no-cache

# Check Docker logs
docker-compose logs

# Verify Dockerfile syntax
docker build --no-cache -t test-image .
```

#### Low Detection Accuracy

**Possible Causes:**
- Model not properly trained
- Image quality issues
- Mismatch between training and inference image sizes
- Confidence threshold too high

**Solutions:**
- Retrain model with more data
- Adjust `MODEL_CONFIDENCE` in `config.py`
- Verify image preprocessing matches training pipeline
- Check model input size matches training (736x736)

For more troubleshooting help, see [DOCKER_TROUBLESHOOTING.md](DOCKER_TROUBLESHOOTING.md)

---

## References

### Technical Documentation

- **YOLO Documentation**: https://docs.ultralytics.com/
- **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **PyTorch Documentation**: https://pytorch.org/docs/

### Research & Standards

- YOLO Object Detection: Redmon et al., "You Only Look Once" (2016)
- Black Sigatoka Disease: Agricultural research on Mycosphaerella fijiensis
- YOLO Format Specification: https://docs.ultralytics.com/datasets/

---

## License

[Specify your license here]

## Contributors

[Add contributor information]

## Acknowledgments

[Add acknowledgments if applicable]

---

**Interactive API docs:** `http://localhost:8000/docs`  
**For feedback/data flow:** see [docs/PHASE_CONNECTION_CHECKLIST.md](docs/PHASE_CONNECTION_CHECKLIST.md) and [docs/ENHANCEMENT_1_IMPLEMENTATION_STATUS.md](docs/ENHANCEMENT_1_IMPLEMENTATION_STATUS.md).
