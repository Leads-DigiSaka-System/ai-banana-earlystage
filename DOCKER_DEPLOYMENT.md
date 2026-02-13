# 🐳 Docker Deployment Guide

## 📋 **Prerequisites**
- Docker installed
- Docker Compose installed (optional but recommended)
- Trained model file (`best.pt`) in `models/weights/` directory

---

## 🚀 **Quick Start**

### **Option 1: Using Docker Compose (Recommended)**

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### **Option 2: Using Docker Directly**

```bash
# Build image
docker build -t banana-disease-api .

# Run container
docker run -d \
  --name banana-disease-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  banana-disease-api

# View logs
docker logs -f banana-disease-api

# Stop
docker stop banana-disease-api
docker rm banana-disease-api
```

---

## 📁 **Directory Structure**

```
ai-banana-earlystage/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── main.py
├── config.py
├── requirements.txt
├── models/
│   └── weights/
│       └── best.pt  ← Your trained model
├── router/
├── services/
└── ...
```

---

## ✅ **Verify Deployment**

### **1. Check Health**
```bash
curl http://localhost:8000/health
```

### **2. Test API**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/classify" \
  -F "file=@test_image.jpg" \
  -F "user_id=test123"
```

### **3. Check API Docs**
Open browser: `http://localhost:8000/docs`

---

## 🔧 **Configuration**

### **Environment Variables (Optional)**

Create `.env` file:
```env
HOST=0.0.0.0
PORT=8000
MODEL_PATH=models/weights/best.pt
```

Update `docker-compose.yml`:
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - HOST=0.0.0.0
  - PORT=8000
```

---

## 📊 **Production Deployment**

### **1. Use GPU (if available)**

Update `Dockerfile`:
```dockerfile
# Use CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# ... rest of Dockerfile
```

Update `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### **2. Add Reverse Proxy (Nginx)**

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

### **3. Scale Services**

```bash
# Run multiple instances
docker-compose up -d --scale api=3
```

---

## 🐛 **Troubleshooting**

### **Issue: Model not found**
```bash
# Check if model exists
ls -lh models/weights/best.pt

# Verify volume mount
docker exec banana-disease-api ls -la /app/models/weights/
```

### **Issue: Port already in use**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### **Issue: Out of memory**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
```

### **View Logs**
```bash
# Docker Compose
docker-compose logs -f api

# Docker
docker logs -f banana-disease-api
```

---

## 📦 **Build for Production**

### **1. Optimize Image Size**
```dockerfile
# Use multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **2. Tag and Push to Registry**
```bash
# Tag image
docker tag banana-disease-api your-registry/banana-disease-api:latest

# Push to registry
docker push your-registry/banana-disease-api:latest
```

---

## ✅ **Checklist**

- [ ] Model file (`best.pt`) in `models/weights/`
- [ ] Docker installed
- [ ] Port 8000 available
- [ ] Test API endpoint
- [ ] Check health endpoint
- [ ] Verify model loading

---

## 🎯 **Next Steps**

1. **Deploy to Cloud:**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform

2. **Add Monitoring:**
   - Prometheus + Grafana
   - Sentry for error tracking
   - Log aggregation

3. **Add CI/CD:**
   - GitHub Actions
   - GitLab CI
   - Jenkins

---

**Ready to deploy!** 🚀

