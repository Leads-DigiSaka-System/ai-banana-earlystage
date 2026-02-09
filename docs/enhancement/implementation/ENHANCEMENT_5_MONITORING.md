# Enhancement 5: Production Monitoring & Analytics

## 📋 Overview

**Enhancement Type:** Observability / Monitoring  
**Priority:** 🟢 LOW (but important for production)  
**Estimated Timeline:** 1-2 weeks  
**Complexity:** Medium  
**Dependencies:** All previous enhancements (especially 1, 2, and 3)

---

## 🎯 Why This Enhancement is Needed

### Current Problem:
Without monitoring, you're flying blind in production:
- ❌ Don't know if API is slow or down
- ❌ Can't see model performance degradation
- ❌ No visibility into data drift
- ❌ Can't detect anomalies
- ❌ No alerts when things go wrong
- ❌ Can't debug production issues
- ❌ No usage analytics

### Real Scenario Without Monitoring:
```
Day 1: Model deployed to production ✅
Day 7: Users complain about slow responses ❓
Day 14: Some predictions are wrong ❓
Day 21: API crashes overnight 🔥
Day 30: You realize accuracy dropped from 85% to 70% 😱

Problem: You only discover issues when users complain!
```

### With Monitoring (This Enhancement):
```
Day 1: Model deployed, dashboards show green ✅
Day 3: Alert: Response time increased by 20% 📊
       → You investigate and optimize
Day 7: Alert: Accuracy dropped to 80% 📉
       → You check data drift, retrain if needed
Day 10: Alert: Error rate spiked to 5% 🚨
        → You rollback deployment automatically
Day 14: Dashboard shows usage patterns 📈
        → You optimize for peak hours

Result: Proactive problem detection and resolution!
```

### Benefits After Implementation:
1. ✅ **Know system health** - Real-time status of all services
2. ✅ **Detect issues early** - Before users complain
3. ✅ **Track model performance** - Real accuracy in production
4. ✅ **Monitor data drift** - Know when to retrain
5. ✅ **Debug faster** - Logs and traces for troubleshooting
6. ✅ **Make data-driven decisions** - Usage analytics
7. ✅ **SLA compliance** - Track uptime and performance

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   Production System                           │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐           │
│  │  FastAPI   │  │  Database  │  │   MLflow    │           │
│  │  (API)     │  │  (Postgres)│  │  (Models)   │           │
│  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘           │
│        │               │                 │                   │
│        └───────────────┴─────────────────┘                   │
│                        │                                     │
│                        │ Metrics, Logs, Traces               │
│                        ▼                                     │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Observability Stack                           │
│                                                               │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Prometheus    │  │     Loki     │  │     Jaeger    │  │
│  │   (Metrics)     │  │    (Logs)    │  │    (Traces)   │  │
│  └────────┬────────┘  └───────┬──────┘  └───────┬───────┘  │
│           │                    │                 │           │
│           └────────────────────┴─────────────────┘           │
│                                │                             │
│                                ▼                             │
│                        ┌───────────────┐                     │
│                        │    Grafana    │                     │
│                        │  (Dashboards) │                     │
│                        └───────┬───────┘                     │
└────────────────────────────────┼─────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                    Alerting & Notifications                   │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Email   │  │  Slack   │  │ PagerDuty│  │   SMS    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Guide

### Phase 1: Metrics Collection (Days 1-3)

#### Step 1.1: Install Prometheus Client

```bash
pip install prometheus-client
```

#### Step 1.2: Add Metrics to FastAPI

Update `main.py`:

```python
# main.py

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
import time
from starlette.responses import Response

app = FastAPI()

# Define metrics

# Request counters
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

prediction_count = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_version', 'predicted_class']
)

feedback_count = Counter(
    'feedback_submissions_total',
    'Total feedback submissions',
    ['is_correct', 'feedback_source']
)

# Histograms for latency
request_latency = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['method', 'endpoint']
)

inference_latency = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    ['model_version']
)

# Gauges for current state
active_requests = Gauge(
    'api_active_requests',
    'Number of active requests'
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy from feedback',
    ['model_version', 'time_window']
)

error_rate = Gauge(
    'api_error_rate',
    'API error rate',
    ['endpoint']
)

# Middleware for automatic metrics collection

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Track active requests
    active_requests.inc()
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_latency.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
        
    except Exception as e:
        # Record error
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
        
    finally:
        active_requests.dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose metrics for Prometheus"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

# Update prediction endpoint to track metrics
@app.post("/api/v1/predict")
async def predict(file: UploadFile, user_id: str):
    start_time = time.time()
    
    # ... existing prediction code ...
    
    # Track metrics
    inference_time = time.time() - start_time
    
    inference_latency.labels(
        model_version=model_version
    ).observe(inference_time)
    
    prediction_count.labels(
        model_version=model_version,
        predicted_class=predicted_class
    ).inc()
    
    return result
```

#### Step 1.3: Setup Prometheus

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus

  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./monitoring/loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    container_name: promtail
    volumes:
      - ./logs:/var/log
      - ./monitoring/promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
```

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'banana-api'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Your FastAPI app
    metrics_path: '/metrics'
```

Start monitoring stack:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

---

### Phase 2: Logging & Tracing (Days 4-5)

#### Step 2.1: Setup Structured Logging

Create `utils/logging_config.py`:

```python
# utils/logging_config.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'prediction_id'):
            log_data['prediction_id'] = record.prediction_id
        if hasattr(record, 'model_version'):
            log_data['model_version'] = record.model_version
        
        return json.dumps(log_data)

def setup_logging():
    """Configure application logging"""
    
    # Create logger
    logger = logging.getLogger('banana_api')
    logger.setLevel(logging.INFO)
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    
    # File handler (JSON for parsing)
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Usage in your app
logger = setup_logging()

# Example usage
logger.info(
    "Prediction made",
    extra={
        'user_id': 'user123',
        'prediction_id': 'pred456',
        'model_version': 'v1.2',
        'confidence': 0.85
    }
)
```

#### Step 2.2: Configure Loki

Create `monitoring/loki-config.yml`:

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

Create `monitoring/promtail-config.yml`:

```yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: banana-api-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: banana-api
          __path__: /var/log/app.log
```

---

### Phase 3: Dashboards (Days 6-8)

#### Step 3.1: Create Grafana Dashboards

Create `monitoring/grafana/dashboards/api_overview.json`:

```json
{
  "dashboard": {
    "title": "Banana Sigatoka API - Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      },
      {
        "title": "Average Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_request_latency_seconds_sum[5m]) / rate(api_request_latency_seconds_count[5m])"
          }
        ]
      },
      {
        "title": "Active Requests",
        "type": "stat",
        "targets": [
          {
            "expr": "api_active_requests"
          }
        ]
      },
      {
        "title": "Predictions by Class",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (predicted_class) (model_predictions_total)"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "model_accuracy"
          }
        ]
      }
    ]
  }
}
```

Create `monitoring/grafana/dashboards/model_performance.json`:

```json
{
  "dashboard": {
    "title": "Model Performance",
    "panels": [
      {
        "title": "Inference Latency (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Predictions per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_predictions_total[1m])"
          }
        ]
      },
      {
        "title": "Feedback Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(feedback_submissions_total[5m])"
          }
        ]
      },
      {
        "title": "Accuracy by Class",
        "type": "table",
        "targets": [
          {
            "expr": "model_accuracy"
          }
        ]
      },
      {
        "title": "Correct vs Incorrect Predictions",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (is_correct) (feedback_submissions_total)"
          }
        ]
      }
    ]
  }
}
```

---

### Phase 4: Alerting (Days 9-10)

#### Step 4.1: Configure Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(api_requests_total{status=~"5.."}[5m]) / rate(api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }} seconds"
      
      # Low accuracy
      - alert: ModelAccuracyDrop
        expr: |
          model_accuracy < 0.75
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy dropped"
          description: "Accuracy is {{ $value | humanizePercentage }}"
      
      # API down
      - alert: APIDown
        expr: |
          up{job="banana-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "API is down"
          description: "API has been down for 2 minutes"

  - name: model_alerts
    interval: 1m
    rules:
      # Data drift detected
      - alert: DataDriftDetected
        expr: |
          data_drift_score > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Drift score is {{ $value }}"
```

#### Step 4.2: Setup Alertmanager

Update `docker-compose.monitoring.yml`:

```yaml
  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'

volumes:
  # ... existing volumes ...
  alertmanager_data:
```

Create `monitoring/alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true
    
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@example.com'
  
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@example.com'
        headers:
          Subject: '[CRITICAL] {{ .GroupLabels.alertname }}'
    
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
  
  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts-warning'
```

---

### Phase 5: Data Drift Detection (Days 11-12)

#### Step 5.1: Implement Drift Detection

Create `monitoring/drift_detection.py`:

```python
# monitoring/drift_detection.py

from scipy import stats
import numpy as np
from database.connection import get_db
from database.models import Prediction
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from services.storage_service import StorageService
from prometheus_client import Gauge

# Metric for data drift
data_drift_score = Gauge(
    'data_drift_score',
    'Data drift score (0-1)',
    ['metric_type']
)

class DriftDetector:
    def __init__(self):
        self.storage = StorageService()
        self.reference_stats = None
    
    def calculate_image_stats(self, image_data: bytes):
        """Calculate statistical properties of an image"""
        img = Image.open(BytesIO(image_data))
        img_array = np.array(img)
        
        return {
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'mean_r': np.mean(img_array[:, :, 0]) if len(img_array.shape) == 3 else 0,
            'mean_g': np.mean(img_array[:, :, 1]) if len(img_array.shape) == 3 else 0,
            'mean_b': np.mean(img_array[:, :, 2]) if len(img_array.shape) == 3 else 0,
        }
    
    def set_reference_distribution(self, days_back: int = 30):
        """
        Set reference distribution from historical data
        
        Args:
            days_back: Number of days to use for reference
        """
        with get_db() as db:
            # Get predictions from reference period
            cutoff = datetime.now() - timedelta(days=days_back)
            predictions = db.query(Prediction).filter(
                Prediction.timestamp >= cutoff
            ).limit(1000).all()  # Sample 1000 images
            
            stats_list = []
            
            for pred in predictions:
                try:
                    image_data = self.storage.get_image(pred.image_path)
                    stats = self.calculate_image_stats(image_data)
                    stats_list.append(stats)
                except:
                    continue
            
            # Calculate reference statistics
            self.reference_stats = {
                'mean_brightness': [s['mean_brightness'] for s in stats_list],
                'std_brightness': [s['std_brightness'] for s in stats_list],
                'mean_r': [s['mean_r'] for s in stats_list],
                'mean_g': [s['mean_g'] for s in stats_list],
                'mean_b': [s['mean_b'] for s in stats_list],
            }
            
            print(f"Reference distribution set with {len(stats_list)} samples")
    
    def detect_drift(self, days: int = 7):
        """
        Detect data drift in recent predictions
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Drift score (0-1), higher = more drift
        """
        if self.reference_stats is None:
            print("No reference distribution set")
            return 0.0
        
        with get_db() as db:
            # Get recent predictions
            cutoff = datetime.now() - timedelta(days=days)
            recent_predictions = db.query(Prediction).filter(
                Prediction.timestamp >= cutoff
            ).limit(500).all()
            
            recent_stats = []
            
            for pred in recent_predictions:
                try:
                    image_data = self.storage.get_image(pred.image_path)
                    stats = self.calculate_image_stats(image_data)
                    recent_stats.append(stats)
                except:
                    continue
            
            if len(recent_stats) < 10:
                print("Not enough recent data for drift detection")
                return 0.0
            
            # Collect current statistics
            current_stats = {
                'mean_brightness': [s['mean_brightness'] for s in recent_stats],
                'std_brightness': [s['std_brightness'] for s in recent_stats],
                'mean_r': [s['mean_r'] for s in recent_stats],
                'mean_g': [s['mean_g'] for s in recent_stats],
                'mean_b': [s['mean_b'] for s in recent_stats],
            }
            
            # Kolmogorov-Smirnov test for each metric
            p_values = []
            
            for metric in self.reference_stats.keys():
                statistic, p_value = stats.ks_2samp(
                    self.reference_stats[metric],
                    current_stats[metric]
                )
                p_values.append(p_value)
            
            # Drift score: 1 - average p-value
            # Lower p-value = more drift
            drift_score = 1 - np.mean(p_values)
            
            # Update Prometheus metric
            data_drift_score.labels(metric_type='image_statistics').set(drift_score)
            
            print(f"Drift score: {drift_score:.4f}")
            
            if drift_score > 0.5:
                print("⚠️ Significant data drift detected! Consider retraining.")
            
            return drift_score

def monitor_drift():
    """Periodic drift monitoring"""
    detector = DriftDetector()
    
    # Set reference from last 30 days
    detector.set_reference_distribution(days_back=30)
    
    # Check drift in last 7 days
    drift_score = detector.detect_drift(days=7)
    
    return drift_score

if __name__ == "__main__":
    monitor_drift()
```

Add to cron or Airflow:
```python
# In Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    'drift_monitoring',
    schedule_interval='0 0 * * *',  # Daily
    catchup=False
)

drift_check = PythonOperator(
    task_id='check_drift',
    python_callable=monitor_drift,
    dag=dag
)
```

---

## 📊 Phase-by-Phase Implementation Schedule

### Phase 1: Metrics (Week 1 - Days 1-3)
- ✅ Install Prometheus client
- ✅ Add metrics to FastAPI
- ✅ Setup Prometheus server
- ✅ Verify metrics collection

### Phase 2: Logging (Week 1 - Days 4-5)
- ✅ Setup structured logging
- ✅ Configure Loki
- ✅ Configure Promtail
- ✅ Verify logs collection

### Phase 3: Dashboards (Week 1-2 - Days 6-8)
- ✅ Create API overview dashboard
- ✅ Create model performance dashboard
- ✅ Create system health dashboard
- ✅ Customize and refine

### Phase 4: Alerting (Week 2 - Days 9-10)
- ✅ Configure alert rules
- ✅ Setup Alertmanager
- ✅ Configure notification channels
- ✅ Test alerts

### Phase 5: Drift Detection (Week 2 - Days 11-12)
- ✅ Implement drift detector
- ✅ Integrate with monitoring
- ✅ Add to Airflow schedule
- ✅ Test and validate

---

## ✅ Verification Checklist

### Metrics
- [ ] Prometheus is scraping metrics
- [ ] Metrics are appearing in Prometheus UI
- [ ] Request metrics are accurate
- [ ] Model metrics are tracked
- [ ] Custom metrics work

### Logging
- [ ] Logs are structured (JSON)
- [ ] Logs appear in Loki
- [ ] Can query logs in Grafana
- [ ] Log levels work correctly
- [ ] Context is preserved

### Dashboards
- [ ] Can access Grafana
- [ ] Dashboards show data
- [ ] Graphs update in real-time
- [ ] All panels work
- [ ] Custom dashboards created

### Alerting
- [ ] Alert rules are loaded
- [ ] Alerts trigger correctly
- [ ] Notifications are sent
- [ ] Alert routing works
- [ ] Can acknowledge/silence alerts

### Drift Detection
- [ ] Drift detector runs
- [ ] Drift scores make sense
- [ ] Alerts on high drift
- [ ] Integrated with pipeline
- [ ] Can trigger retraining

---

## 📈 Key Metrics to Monitor

### API Health
- Request rate (requests/sec)
- Error rate (%)
- Response time (P50, P95, P99)
- Active connections
- Uptime (%)

### Model Performance
- Predictions per second
- Inference latency (ms)
- Accuracy from feedback (%)
- Confidence distribution
- Class distribution

### System Resources
- CPU usage (%)
- Memory usage (MB)
- Disk usage (GB)
- Network I/O (MB/s)

### Business Metrics
- Daily active users
- Total predictions
- Feedback rate (%)
- User retention

---

## 🎓 Next Steps

After implementing monitoring:

1. **Set SLAs**: Define uptime and performance targets
2. **Create runbooks**: Document how to respond to alerts
3. **Regular reviews**: Weekly review of dashboards
4. **Optimize based on data**: Use analytics to improve
5. **Expand monitoring**: Add more metrics as needed

---

## 📋 Sample Alert Responses

### High Error Rate Alert
```
1. Check recent deployments (rollback if needed)
2. Check logs for error patterns
3. Check database connection
4. Check external dependencies
5. Scale up if capacity issue
```

### Model Accuracy Drop Alert
```
1. Check data drift metrics
2. Review recent feedback
3. Check if specific classes affected
4. Consider emergency retraining
5. Investigate data quality issues
```

### High Latency Alert
```
1. Check system resources
2. Check database performance
3. Review recent code changes
4. Check for N+1 queries
5. Consider caching or optimization
```

---

**You can't improve what you don't measure!**
