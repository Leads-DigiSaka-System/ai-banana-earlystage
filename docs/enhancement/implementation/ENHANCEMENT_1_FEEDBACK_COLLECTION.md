# Enhancement 1: Feedback Collection & Data Storage System

## 📋 Overview

**Enhancement Type:** Foundation / Infrastructure  
**Priority:** 🔴 HIGH  
**Estimated Timeline:** 2-3 weeks  
**Complexity:** Medium  
**Dependencies:** None (Can start immediately)

---

## 🎯 Why This Enhancement is Needed

### Current Problem:
Your current system has **no way to know if predictions are correct or wrong**. Once the model makes a prediction, you have:
- ❌ No feedback from users about accuracy
- ❌ No data about real-world performance
- ❌ No mechanism to collect new training data from production
- ❌ No way to identify which classes are problematic
- ❌ No continuous improvement capability

### Impact of Not Having This:
1. **Model becomes stale** - Real-world data changes, but model doesn't improve
2. **Can't measure real accuracy** - Test set accuracy ≠ production accuracy
3. **Can't retrain effectively** - No new labeled data for retraining
4. **User trust decreases** - Errors persist without being fixed
5. **Manual work required** - Need to manually collect and label new images

### Benefits After Implementation:
1. ✅ **Know real-world accuracy** - Track how well model performs in production
2. ✅ **Collect labeled data automatically** - Users provide labels through feedback
3. ✅ **Identify problem areas** - See which disease stages are misclassified
4. ✅ **Enable continuous learning** - Foundation for automated retraining
5. ✅ **Build trust** - Show users that errors are being tracked and fixed

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mobile/Web App                            │
│                                                                  │
│  1. User uploads image                                          │
│  2. Gets prediction: "Stage2, 85% confidence"                   │
│  3. User clicks: ✓ Correct  OR  ✗ Wrong → "Actually Stage3"   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ HTTP POST /api/v1/feedback/submit
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Feedback Service                                        │  │
│  │  • Validate feedback data                                │  │
│  │  • Store in database                                     │  │
│  │  • Save image to object storage                          │  │
│  │  • Queue for retraining if needed                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────┬────────────────────────┬──────────────────────────────┬───┘
     │                        │                              │
     │                        │                              │
     ▼                        ▼                              ▼
┌─────────────┐      ┌─────────────────┐         ┌──────────────────┐
│  PostgreSQL │      │  Object Storage │         │  Training Queue  │
│   Database  │      │   (MinIO/S3)    │         │   (RabbitMQ)     │
│             │      │                 │         │                  │
│ • predictions│      │ • Original images│        │ • Images to      │
│ • feedback  │      │ • Corrected labels│       │   retrain        │
│ • metrics   │      │ • Metadata      │         │                  │
└─────────────┘      └─────────────────┘         └──────────────────┘
```

---

## 📊 Database Schema Design

### Table 1: predictions

Stores every prediction made by the model.

```sql
CREATE TABLE predictions (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- User information
    user_id VARCHAR(255) NOT NULL,
    user_location VARCHAR(255),  -- Optional: geographic data
    
    -- Image information
    image_path VARCHAR(500) NOT NULL,  -- Path in object storage
    image_size_kb INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    image_hash VARCHAR(64),  -- For deduplication
    
    -- Prediction results
    predicted_class_id INTEGER NOT NULL,
    predicted_class_name VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_data JSONB,  -- Bounding box coordinates
    
    -- Model information
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    
    -- Timing
    inference_time_ms FLOAT,  -- How long prediction took
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB,  -- Any additional data
    
    -- Indexes for fast queries
    INDEX idx_user_timestamp (user_id, timestamp),
    INDEX idx_predicted_class (predicted_class_name),
    INDEX idx_model_version (model_version),
    INDEX idx_timestamp (timestamp DESC)
);
```

### Table 2: feedback

Stores user feedback on predictions.

```sql
CREATE TABLE feedback (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to prediction
    prediction_id UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    
    -- Feedback data
    is_correct BOOLEAN NOT NULL,
    correct_class_id INTEGER,  -- If wrong, what's the correct class?
    correct_class_name VARCHAR(50),
    
    -- Additional feedback
    user_comment TEXT,  -- Optional user notes
    confidence_rating INTEGER CHECK (confidence_rating BETWEEN 1 AND 5),
    
    -- Metadata
    feedback_source VARCHAR(50),  -- 'mobile_app', 'web', 'expert_review'
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Processing status
    processed_for_training BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP,
    
    -- Indexes
    INDEX idx_prediction_id (prediction_id),
    INDEX idx_is_correct (is_correct),
    INDEX idx_processed (processed_for_training),
    INDEX idx_timestamp (timestamp DESC)
);
```

### Table 3: training_data

Stores images and labels for retraining.

```sql
CREATE TABLE training_data (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Image information
    image_path VARCHAR(500) NOT NULL,
    image_hash VARCHAR(64) UNIQUE,  -- Prevent duplicates
    
    -- Label information
    class_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    bbox_data JSONB,  -- Bounding box annotations
    
    -- Source tracking
    source VARCHAR(50) NOT NULL,  -- 'original', 'feedback', 'app', 'manual'
    source_id UUID,  -- Link to feedback or prediction
    
    -- Quality metrics
    quality_score FLOAT,  -- Image quality (0-1)
    blur_score FLOAT,
    brightness_score FLOAT,
    
    -- Validation status
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by VARCHAR(255),
    validated_at TIMESTAMP,
    
    -- Dataset assignment
    dataset_split VARCHAR(20),  -- 'train', 'val', 'test', 'pending'
    
    -- Metadata
    added_date TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB,
    
    -- Indexes
    INDEX idx_class_name (class_name),
    INDEX idx_source (source),
    INDEX idx_validated (is_validated),
    INDEX idx_dataset_split (dataset_split),
    INDEX idx_added_date (added_date DESC)
);
```

### Table 4: model_performance

Tracks model performance over time from feedback.

```sql
CREATE TABLE model_performance (
    -- Primary identification
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Model information
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    
    -- Aggregated metrics (calculated daily)
    date DATE NOT NULL,
    
    -- Overall metrics
    total_predictions INTEGER NOT NULL,
    total_feedback INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    incorrect_predictions INTEGER NOT NULL,
    accuracy FLOAT,  -- correct / total_feedback
    
    -- Per-class metrics (JSONB for flexibility)
    class_metrics JSONB,
    /* Example structure:
    {
        "Healthy": {"predictions": 100, "correct": 85, "accuracy": 0.85},
        "Stage1": {"predictions": 50, "correct": 40, "accuracy": 0.80},
        ...
    }
    */
    
    -- Confidence metrics
    avg_confidence FLOAT,
    avg_confidence_correct FLOAT,
    avg_confidence_incorrect FLOAT,
    
    -- Timestamps
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Indexes
    UNIQUE INDEX idx_model_date (model_version, date),
    INDEX idx_date (date DESC)
);
```

---

## 🔧 Implementation Guide

### Phase 1: Database Setup (Days 1-2)

#### Step 1.1: Install PostgreSQL

**Option A: Using Docker (Recommended for Development)**

```bash
# Create docker-compose.yml for database
cat > docker-compose.db.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: banana_sigatoka_db
    environment:
      POSTGRES_DB: banana_sigatoka
      POSTGRES_USER: banana_user
      POSTGRES_PASSWORD: your_secure_password_here
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U banana_user"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
EOF

# Start database
docker-compose -f docker-compose.db.yml up -d

# Check logs
docker-compose -f docker-compose.db.yml logs -f
```

**Option B: Local Installation**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
-- Inside PostgreSQL console
CREATE DATABASE banana_sigatoka;
CREATE USER banana_user WITH PASSWORD 'your_secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE banana_sigatoka TO banana_user;
\q
```

#### Step 1.2: Create Database Schema

Create `init_db.sql`:

```sql
-- init_db.sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    user_location VARCHAR(255),
    image_path VARCHAR(500) NOT NULL,
    image_size_kb INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    image_hash VARCHAR(64),
    predicted_class_id INTEGER NOT NULL,
    predicted_class_name VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_data JSONB,
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    inference_time_ms FLOAT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_predictions_user_timestamp ON predictions(user_id, timestamp);
CREATE INDEX idx_predictions_class ON predictions(predicted_class_name);
CREATE INDEX idx_predictions_model ON predictions(model_version);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp DESC);

-- Create feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    is_correct BOOLEAN NOT NULL,
    correct_class_id INTEGER,
    correct_class_name VARCHAR(50),
    user_comment TEXT,
    confidence_rating INTEGER CHECK (confidence_rating BETWEEN 1 AND 5),
    feedback_source VARCHAR(50),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    processed_for_training BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP
);

CREATE INDEX idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX idx_feedback_correct ON feedback(is_correct);
CREATE INDEX idx_feedback_processed ON feedback(processed_for_training);
CREATE INDEX idx_feedback_timestamp ON feedback(timestamp DESC);

-- Create training_data table
CREATE TABLE training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_path VARCHAR(500) NOT NULL,
    image_hash VARCHAR(64) UNIQUE,
    class_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    bbox_data JSONB,
    source VARCHAR(50) NOT NULL,
    source_id UUID,
    quality_score FLOAT,
    blur_score FLOAT,
    brightness_score FLOAT,
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by VARCHAR(255),
    validated_at TIMESTAMP,
    dataset_split VARCHAR(20),
    added_date TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_training_class ON training_data(class_name);
CREATE INDEX idx_training_source ON training_data(source);
CREATE INDEX idx_training_validated ON training_data(is_validated);
CREATE INDEX idx_training_split ON training_data(dataset_split);
CREATE INDEX idx_training_date ON training_data(added_date DESC);

-- Create model_performance table
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_predictions INTEGER NOT NULL,
    total_feedback INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    incorrect_predictions INTEGER NOT NULL,
    accuracy FLOAT,
    class_metrics JSONB,
    avg_confidence FLOAT,
    avg_confidence_correct FLOAT,
    avg_confidence_incorrect FLOAT,
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_performance_model_date ON model_performance(model_version, date);
CREATE INDEX idx_performance_date ON model_performance(date DESC);
```

Execute the schema:

```bash
# If using Docker
docker exec -i banana_sigatoka_db psql -U banana_user -d banana_sigatoka < init_db.sql

# If using local PostgreSQL
psql -U banana_user -d banana_sigatoka -f init_db.sql
```

#### Step 1.3: Install Database Dependencies

```bash
# Add to requirements.txt
cat >> requirements.txt << 'EOF'
# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.13.1  # For database migrations

# Object Storage
minio==7.2.0
boto3==1.34.0  # For AWS S3 compatibility

# Message Queue (optional for now)
# pika==1.3.2  # RabbitMQ client
EOF

# Install dependencies
pip install -r requirements.txt
```

---

### Phase 2: Object Storage Setup (Days 2-3)

#### Step 2.1: Install MinIO (S3-compatible storage)

**Using Docker:**

```bash
# Add to docker-compose.db.yml
cat >> docker-compose.db.yml << 'EOF'

  minio:
    image: minio/minio:latest
    container_name: banana_sigatoka_minio
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  postgres_data:
  minio_data:
EOF

# Restart docker-compose
docker-compose -f docker-compose.db.yml up -d
```

Access MinIO Console at `http://localhost:9001` (user: minioadmin, password: minioadmin123)

#### Step 2.2: Create Storage Client

Create `services/storage_service.py`:

```python
# services/storage_service.py

from minio import Minio
from minio.error import S3Error
from pathlib import Path
import hashlib
from datetime import timedelta
from typing import Optional
import io
from PIL import Image

class StorageService:
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin123",
        secure: bool = False
    ):
        """
        Initialize MinIO storage client
        
        Args:
            endpoint: MinIO server endpoint
            access_key: Access key
            secret_key: Secret key
            secure: Use HTTPS (True) or HTTP (False)
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        # Bucket names
        self.predictions_bucket = "predictions"
        self.training_bucket = "training-data"
        self.models_bucket = "models"
        
        # Create buckets if they don't exist
        self._ensure_buckets_exist()
    
    def _ensure_buckets_exist(self):
        """Create buckets if they don't exist"""
        buckets = [
            self.predictions_bucket,
            self.training_bucket,
            self.models_bucket
        ]
        
        for bucket in buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                print(f"Created bucket: {bucket}")
    
    def upload_prediction_image(
        self,
        image_data: bytes,
        user_id: str,
        prediction_id: str,
        file_extension: str = "jpg"
    ) -> str:
        """
        Upload prediction image to storage
        
        Args:
            image_data: Image bytes
            user_id: User identifier
            prediction_id: Prediction UUID
            file_extension: File extension
            
        Returns:
            Path to uploaded image
        """
        # Generate image hash for deduplication
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Create path: predictions/{user_id}/{year}/{month}/{prediction_id}_{hash}.jpg
        from datetime import datetime
        now = datetime.now()
        object_name = (
            f"{user_id}/"
            f"{now.year}/"
            f"{now.month:02d}/"
            f"{prediction_id}_{image_hash[:8]}.{file_extension}"
        )
        
        # Upload to MinIO
        self.client.put_object(
            bucket_name=self.predictions_bucket,
            object_name=object_name,
            data=io.BytesIO(image_data),
            length=len(image_data),
            content_type=f"image/{file_extension}"
        )
        
        # Return path
        return f"{self.predictions_bucket}/{object_name}"
    
    def upload_training_image(
        self,
        image_data: bytes,
        class_name: str,
        image_hash: str,
        file_extension: str = "jpg"
    ) -> str:
        """
        Upload training image to storage
        
        Args:
            image_data: Image bytes
            class_name: Class label
            image_hash: Image hash (for filename)
            file_extension: File extension
            
        Returns:
            Path to uploaded image
        """
        # Create path: training-data/{class_name}/{hash}.jpg
        object_name = f"{class_name}/{image_hash}.{file_extension}"
        
        # Upload to MinIO
        self.client.put_object(
            bucket_name=self.training_bucket,
            object_name=object_name,
            data=io.BytesIO(image_data),
            length=len(image_data),
            content_type=f"image/{file_extension}"
        )
        
        return f"{self.training_bucket}/{object_name}"
    
    def get_image(self, image_path: str) -> bytes:
        """
        Retrieve image from storage
        
        Args:
            image_path: Full path (bucket/object)
            
        Returns:
            Image bytes
        """
        # Parse bucket and object name
        parts = image_path.split('/', 1)
        bucket_name = parts[0]
        object_name = parts[1]
        
        # Download from MinIO
        response = self.client.get_object(bucket_name, object_name)
        image_data = response.read()
        response.close()
        response.release_conn()
        
        return image_data
    
    def delete_image(self, image_path: str):
        """Delete image from storage"""
        parts = image_path.split('/', 1)
        bucket_name = parts[0]
        object_name = parts[1]
        
        self.client.remove_object(bucket_name, object_name)
    
    def get_presigned_url(
        self,
        image_path: str,
        expires: timedelta = timedelta(hours=1)
    ) -> str:
        """
        Get presigned URL for image access
        
        Args:
            image_path: Full path (bucket/object)
            expires: URL expiration time
            
        Returns:
            Presigned URL
        """
        parts = image_path.split('/', 1)
        bucket_name = parts[0]
        object_name = parts[1]
        
        url = self.client.presigned_get_object(
            bucket_name,
            object_name,
            expires=expires
        )
        
        return url
```

Create `config.py` for storage configuration:

```python
# config.py (update existing file)

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Existing settings...
    MODEL_PATH: Path = Path("models/weights/best.pt")
    
    # Database settings
    DATABASE_URL: str = "postgresql://banana_user:your_secure_password_here@localhost:5432/banana_sigatoka"
    
    # Storage settings
    STORAGE_ENDPOINT: str = "localhost:9000"
    STORAGE_ACCESS_KEY: str = "minioadmin"
    STORAGE_SECRET_KEY: str = "minioadmin123"
    STORAGE_SECURE: bool = False
    
    # API settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

### Phase 3: Database Service Layer (Days 3-4)

#### Step 3.1: Create Database Models

Create `database/models.py`:

```python
# database/models.py

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    DateTime, ForeignKey, JSON, Date, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    user_location = Column(String(255))
    
    image_path = Column(String(500), nullable=False)
    image_size_kb = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    image_hash = Column(String(64))
    
    predicted_class_id = Column(Integer, nullable=False)
    predicted_class_name = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_data = Column(JSON)
    
    model_version = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    
    inference_time_ms = Column(Float)
    timestamp = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    metadata = Column(JSON)

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey('predictions.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    is_correct = Column(Boolean, nullable=False, index=True)
    correct_class_id = Column(Integer)
    correct_class_name = Column(String(50))
    
    user_comment = Column(String)
    confidence_rating = Column(Integer)
    
    feedback_source = Column(String(50))
    timestamp = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    
    processed_for_training = Column(Boolean, default=False, index=True)
    processed_at = Column(DateTime)
    
    __table_args__ = (
        CheckConstraint('confidence_rating BETWEEN 1 AND 5'),
    )

class TrainingData(Base):
    __tablename__ = 'training_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), unique=True)
    
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False, index=True)
    bbox_data = Column(JSON)
    
    source = Column(String(50), nullable=False, index=True)
    source_id = Column(UUID(as_uuid=True))
    
    quality_score = Column(Float)
    blur_score = Column(Float)
    brightness_score = Column(Float)
    
    is_validated = Column(Boolean, default=False, index=True)
    validated_by = Column(String(255))
    validated_at = Column(DateTime)
    
    dataset_split = Column(String(20), index=True)
    added_date = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    metadata = Column(JSON)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    model_version = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    total_predictions = Column(Integer, nullable=False)
    total_feedback = Column(Integer, nullable=False)
    correct_predictions = Column(Integer, nullable=False)
    incorrect_predictions = Column(Integer, nullable=False)
    accuracy = Column(Float)
    
    class_metrics = Column(JSON)
    
    avg_confidence = Column(Float)
    avg_confidence_correct = Column(Float)
    avg_confidence_incorrect = Column(Float)
    
    calculated_at = Column(DateTime, nullable=False, server_default=func.now())
```

#### Step 3.2: Create Database Connection

Create `database/connection.py`:

```python
# database/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from config import settings
from database.models import Base

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

@contextmanager
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Dependency for FastAPI
def get_db_dependency():
    """FastAPI dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### Phase 4: Feedback API Implementation (Days 4-6)

#### Step 4.1: Create Data Collection Service

Create `services/data_collection_service.py`:

```python
# services/data_collection_service.py

from sqlalchemy.orm import Session
from database.models import Prediction, Feedback, TrainingData
from services.storage_service import StorageService
from typing import Optional, Dict, Any
import hashlib
from datetime import datetime
import uuid

class DataCollectionService:
    def __init__(self, db: Session, storage: StorageService):
        self.db = db
        self.storage = storage
    
    async def save_prediction(
        self,
        user_id: str,
        image_data: bytes,
        prediction_result: Dict[str, Any],
        model_version: str,
        inference_time_ms: float,
        user_location: Optional[str] = None
    ) -> str:
        """
        Save prediction to database and storage
        
        Args:
            user_id: User identifier
            image_data: Image bytes
            prediction_result: Prediction results from model
            model_version: Model version used
            inference_time_ms: Inference time
            user_location: Optional user location
            
        Returns:
            Prediction ID (UUID)
        """
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Calculate image hash
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Upload image to storage
        image_path = self.storage.upload_prediction_image(
            image_data=image_data,
            user_id=user_id,
            prediction_id=prediction_id,
            file_extension="jpg"
        )
        
        # Get image dimensions
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_data))
        image_width, image_height = img.size
        image_size_kb = len(image_data) // 1024
        
        # Create prediction record
        prediction = Prediction(
            id=prediction_id,
            user_id=user_id,
            user_location=user_location,
            image_path=image_path,
            image_size_kb=image_size_kb,
            image_width=image_width,
            image_height=image_height,
            image_hash=image_hash,
            predicted_class_id=prediction_result.get('class_id'),
            predicted_class_name=prediction_result.get('class_name'),
            confidence=prediction_result.get('confidence'),
            bbox_data=prediction_result.get('bbox'),
            model_version=model_version,
            model_name="yolo12n",
            inference_time_ms=inference_time_ms
        )
        
        self.db.add(prediction)
        self.db.commit()
        
        return prediction_id
    
    async def save_feedback(
        self,
        prediction_id: str,
        is_correct: bool,
        correct_class_name: Optional[str] = None,
        correct_class_id: Optional[int] = None,
        user_comment: Optional[str] = None,
        confidence_rating: Optional[int] = None,
        feedback_source: str = "mobile_app"
    ) -> str:
        """
        Save user feedback
        
        Args:
            prediction_id: Prediction UUID
            is_correct: Is prediction correct?
            correct_class_name: Correct class if wrong
            correct_class_id: Correct class ID if wrong
            user_comment: Optional comment
            confidence_rating: User confidence (1-5)
            feedback_source: Source of feedback
            
        Returns:
            Feedback ID (UUID)
        """
        # Create feedback record
        feedback = Feedback(
            prediction_id=prediction_id,
            is_correct=is_correct,
            correct_class_id=correct_class_id,
            correct_class_name=correct_class_name,
            user_comment=user_comment,
            confidence_rating=confidence_rating,
            feedback_source=feedback_source
        )
        
        self.db.add(feedback)
        self.db.commit()
        
        # If incorrect, add to training queue
        if not is_correct and correct_class_name:
            await self.queue_for_retraining(
                prediction_id=prediction_id,
                correct_class_name=correct_class_name,
                correct_class_id=correct_class_id
            )
        
        return str(feedback.id)
    
    async def queue_for_retraining(
        self,
        prediction_id: str,
        correct_class_name: str,
        correct_class_id: int
    ):
        """
        Add corrected prediction to training queue
        
        Args:
            prediction_id: Prediction UUID
            correct_class_name: Correct class name
            correct_class_id: Correct class ID
        """
        # Get prediction
        prediction = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not prediction:
            return
        
        # Check if already in training data
        existing = self.db.query(TrainingData).filter(
            TrainingData.image_hash == prediction.image_hash
        ).first()
        
        if existing:
            # Update existing record
            existing.class_name = correct_class_name
            existing.class_id = correct_class_id
            existing.source = "feedback"
            existing.source_id = prediction_id
        else:
            # Create new training data record
            training_data = TrainingData(
                image_path=prediction.image_path,
                image_hash=prediction.image_hash,
                class_id=correct_class_id,
                class_name=correct_class_name,
                bbox_data=prediction.bbox_data,
                source="feedback",
                source_id=prediction_id,
                dataset_split="pending"  # Will be assigned during preprocessing
            )
            self.db.add(training_data)
        
        self.db.commit()
    
    def get_feedback_statistics(
        self,
        model_version: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get feedback statistics
        
        Args:
            model_version: Filter by model version
            days: Number of days to look back
            
        Returns:
            Statistics dictionary
        """
        from datetime import timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query = self.db.query(Prediction).filter(
            Prediction.timestamp >= start_date
        )
        
        if model_version:
            query = query.filter(Prediction.model_version == model_version)
        
        predictions = query.all()
        
        # Get feedback for these predictions
        prediction_ids = [p.id for p in predictions]
        feedbacks = self.db.query(Feedback).filter(
            Feedback.prediction_id.in_(prediction_ids)
        ).all()
        
        # Calculate statistics
        total_predictions = len(predictions)
        total_feedback = len(feedbacks)
        correct_feedback = sum(1 for f in feedbacks if f.is_correct)
        incorrect_feedback = total_feedback - correct_feedback
        
        accuracy = correct_feedback / total_feedback if total_feedback > 0 else 0
        
        # Per-class statistics
        class_stats = {}
        for prediction in predictions:
            class_name = prediction.predicted_class_name
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'total_predictions': 0,
                    'total_feedback': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'accuracy': 0
                }
            
            class_stats[class_name]['total_predictions'] += 1
        
        # Add feedback to class stats
        for feedback in feedbacks:
            prediction = next((p for p in predictions if p.id == feedback.prediction_id), None)
            if prediction:
                class_name = prediction.predicted_class_name
                class_stats[class_name]['total_feedback'] += 1
                if feedback.is_correct:
                    class_stats[class_name]['correct'] += 1
                else:
                    class_stats[class_name]['incorrect'] += 1
        
        # Calculate per-class accuracy
        for class_name in class_stats:
            total = class_stats[class_name]['total_feedback']
            if total > 0:
                class_stats[class_name]['accuracy'] = (
                    class_stats[class_name]['correct'] / total
                )
        
        return {
            'period_days': days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_predictions': total_predictions,
            'total_feedback': total_feedback,
            'correct_predictions': correct_feedback,
            'incorrect_predictions': incorrect_feedback,
            'overall_accuracy': accuracy,
            'feedback_rate': total_feedback / total_predictions if total_predictions > 0 else 0,
            'class_statistics': class_stats
        }
```

#### Step 4.2: Update FastAPI Router

Update `router/process.py` to include feedback endpoints:

```python
# router/process.py (add these endpoints)

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from database.connection import get_db_dependency
from services.data_collection_service import DataCollectionService
from services.storage_service import StorageService
from services.detection_service import DetectionService
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Initialize storage service (singleton)
storage_service = StorageService()

class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    correct_class_name: Optional[str] = None
    correct_class_id: Optional[int] = None
    user_comment: Optional[str] = None
    confidence_rating: Optional[int] = None

@router.post("/predict")
async def predict_with_tracking(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    user_location: Optional[str] = Form(None),
    db: Session = Depends(get_db_dependency)
):
    """
    Make prediction and save to database
    """
    # Read image
    image_data = await file.read()
    
    # Start timer
    start_time = time.time()
    
    # Run prediction (use your existing detection service)
    detection_service = DetectionService()
    prediction_result = detection_service.predict(image_data)
    
    # Calculate inference time
    inference_time_ms = (time.time() - start_time) * 1000
    
    # Save to database
    data_service = DataCollectionService(db, storage_service)
    prediction_id = await data_service.save_prediction(
        user_id=user_id,
        image_data=image_data,
        prediction_result=prediction_result,
        model_version="v1.0.0",  # Get from your model
        inference_time_ms=inference_time_ms,
        user_location=user_location
    )
    
    # Return result with prediction ID
    return {
        "success": True,
        "prediction_id": prediction_id,
        "result": prediction_result,
        "inference_time_ms": inference_time_ms
    }

@router.post("/feedback/submit")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Submit feedback on a prediction
    """
    data_service = DataCollectionService(db, storage_service)
    
    try:
        feedback_id = await data_service.save_feedback(
            prediction_id=feedback.prediction_id,
            is_correct=feedback.is_correct,
            correct_class_name=feedback.correct_class_name,
            correct_class_id=feedback.correct_class_id,
            user_comment=feedback.user_comment,
            confidence_rating=feedback.confidence_rating
        )
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/feedback/stats")
async def get_feedback_stats(
    days: int = 7,
    model_version: Optional[str] = None,
    db: Session = Depends(get_db_dependency)
):
    """
    Get feedback statistics
    """
    data_service = DataCollectionService(db, storage_service)
    stats = data_service.get_feedback_statistics(
        model_version=model_version,
        days=days
    )
    
    return stats
```

---

### Phase 5: Testing & Validation (Days 6-7)

#### Step 5.1: Create Test Script

Create `tests/test_feedback_system.py`:

```python
# tests/test_feedback_system.py

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"

def test_prediction_with_feedback():
    """Test complete workflow: prediction → feedback"""
    
    # 1. Make a prediction
    print("1. Making prediction...")
    
    # Use a sample image
    image_path = "test_images/sample_leaf.jpg"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'user_id': 'test_user_001',
            'user_location': 'Navotas, Philippines'
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            files=files,
            data=data
        )
    
    result = response.json()
    print(f"Prediction result: {json.dumps(result, indent=2)}")
    
    prediction_id = result['prediction_id']
    predicted_class = result['result']['class_name']
    
    # 2. Submit positive feedback (correct prediction)
    print("\n2. Submitting positive feedback...")
    
    feedback_data = {
        "prediction_id": prediction_id,
        "is_correct": True,
        "confidence_rating": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/feedback/submit",
        json=feedback_data
    )
    
    print(f"Feedback response: {json.dumps(response.json(), indent=2)}")
    
    # 3. Make another prediction with wrong result
    print("\n3. Making another prediction for negative feedback test...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'user_id': 'test_user_002'}
        
        response = requests.post(
            f"{BASE_URL}/predict",
            files=files,
            data=data
        )
    
    result = response.json()
    prediction_id_2 = result['prediction_id']
    
    # 4. Submit negative feedback (wrong prediction)
    print("\n4. Submitting negative feedback...")
    
    feedback_data = {
        "prediction_id": prediction_id_2,
        "is_correct": False,
        "correct_class_name": "Stage3",
        "correct_class_id": 3,
        "user_comment": "This is clearly Stage3, not what you predicted",
        "confidence_rating": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/feedback/submit",
        json=feedback_data
    )
    
    print(f"Feedback response: {json.dumps(response.json(), indent=2)}")
    
    # 5. Get feedback statistics
    print("\n5. Getting feedback statistics...")
    
    response = requests.get(f"{BASE_URL}/feedback/stats?days=7")
    stats = response.json()
    
    print(f"Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    test_prediction_with_feedback()
```

Run the test:

```bash
python tests/test_feedback_system.py
```

---

## 📊 Phase-by-Phase Implementation Schedule

### Phase 1: Foundation (Week 1)
**Goal:** Set up infrastructure

**Days 1-2: Database Setup**
- ✅ Install PostgreSQL
- ✅ Create database schema
- ✅ Test database connection
- ✅ Create sample queries

**Days 2-3: Object Storage**
- ✅ Install MinIO
- ✅ Create buckets
- ✅ Test file upload/download
- ✅ Implement storage service

**Days 3-4: Database Layer**
- ✅ Create SQLAlchemy models
- ✅ Implement database connection
- ✅ Create service layer
- ✅ Write unit tests

### Phase 2: API Implementation (Week 2)
**Goal:** Build feedback collection system

**Days 4-5: Data Collection Service**
- ✅ Implement prediction saving
- ✅ Implement feedback saving
- ✅ Implement statistics calculation
- ✅ Add error handling

**Days 5-6: API Endpoints**
- ✅ Update prediction endpoint
- ✅ Create feedback endpoint
- ✅ Create statistics endpoint
- ✅ Add authentication (optional)

**Day 6-7: Testing**
- ✅ Write integration tests
- ✅ Test end-to-end workflow
- ✅ Load testing
- ✅ Fix bugs

### Phase 3: Documentation & Deployment (Week 3)
**Goal:** Deploy and document

**Days 8-9: Docker Integration**
- ✅ Update Dockerfile
- ✅ Update docker-compose.yml
- ✅ Test containerized deployment
- ✅ Write deployment guide

**Day 10: Documentation**
- ✅ API documentation
- ✅ User guide
- ✅ Admin guide
- ✅ Troubleshooting guide

---

## ✅ Verification Checklist

### Database
- [ ] PostgreSQL is running
- [ ] All tables are created
- [ ] Indexes are created
- [ ] Can insert/query data
- [ ] Foreign keys are working

### Storage
- [ ] MinIO is running
- [ ] Buckets are created
- [ ] Can upload images
- [ ] Can download images
- [ ] Can generate presigned URLs

### API
- [ ] Prediction endpoint saves to database
- [ ] Feedback endpoint works
- [ ] Statistics endpoint returns data
- [ ] Error handling works
- [ ] API documentation is updated

### Integration
- [ ] End-to-end test passes
- [ ] Images are stored correctly
- [ ] Database records are created
- [ ] Feedback is linked to predictions
- [ ] Training queue is populated

---

## 📈 Success Metrics

After implementing this enhancement, you should be able to:

1. **Track every prediction** - See all predictions in database
2. **Collect user feedback** - Users can mark predictions as correct/wrong
3. **Calculate real accuracy** - Know model's true performance
4. **Identify problem areas** - See which classes need improvement
5. **Build training dataset** - Have labeled data for retraining

**Target Metrics:**
- Feedback collection rate: **> 70%** of predictions
- Database query time: **< 100ms** for most queries
- Image upload time: **< 500ms** per image
- API response time: **< 200ms** (excluding model inference)
- Data quality: **> 90%** of feedback is valid

---

## 🚨 Common Issues & Solutions

### Issue 1: Database Connection Errors

**Symptoms:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check PostgreSQL logs
docker logs banana_sigatoka_db

# Restart PostgreSQL
docker-compose -f docker-compose.db.yml restart postgres

# Verify connection
psql -U banana_user -d banana_sigatoka -h localhost
```

### Issue 2: MinIO Connection Errors

**Symptoms:**
```
minio.error.S3Error: Unable to connect to MinIO
```

**Solutions:**
```bash
# Check if MinIO is running
docker ps | grep minio

# Access MinIO console
# http://localhost:9001

# Check buckets exist
python -c "
from services.storage_service import StorageService
storage = StorageService()
print('Buckets exist!')
"
```

### Issue 3: Large Image Upload Failures

**Symptoms:**
```
413 Request Entity Too Large
```

**Solutions:**
```python
# Update config.py
MAX_IMAGE_SIZE_MB = 50

# In main.py, add:
from fastapi import FastAPI
app = FastAPI()

@app.middleware("http")
async def limit_upload_size(request, call_next):
    if request.headers.get("content-length"):
        size = int(request.headers["content-length"])
        if size > 50 * 1024 * 1024:  # 50MB
            return JSONResponse(
                status_code=413,
                content={"error": "File too large"}
            )
    return await call_next(request)
```

---

## 🎓 Next Steps

After completing this enhancement:

1. **Move to Enhancement 2**: MLOps Pipeline for automated retraining
2. **Integrate with mobile app**: Update app to send feedback
3. **Monitor metrics**: Track feedback collection rate
4. **Analyze feedback**: Identify which classes need improvement
5. **Plan retraining**: Once you have 500+ feedback samples

---

**This enhancement is the foundation for continuous improvement. Without it, your model will never get better!**
