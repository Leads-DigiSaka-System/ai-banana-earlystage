from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from router.process import router
from router.feedback import router as feedback_router
from router.database_crud import router as db_router
from router.mlops import router as mlops_router
from services.inference import load_model
from config import HOST, PORT, CLASS_NAMES

# Initialize FastAPI app
app = FastAPI(
    title="Banana Disease Detection API",
    description="API for detecting Black Sigatoka disease stages in banana leaves",
    version="1.0.0"
)

# Enable CORS (for web/mobile clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)
app.include_router(feedback_router)
app.include_router(db_router)
app.include_router(mlops_router)


@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    try:
        load_model()
        print("✅ API server ready!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model: {e}")
        print("   Model will be loaded on first request.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Banana Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/api/v1/predict": "POST - Upload image for disease detection (with bounding boxes)",
            "/api/v1/predict/classify": "POST - Classification only (no bounding boxes)",
            "/api/v1/feedback": "Feedback: POST /submit, GET /stats - see /docs#/feedback",
            "/api/v1/db/feedback": "GET list, GET /:id, PATCH /:id, DELETE /:id - CRUD",
            "/api/v1/db/predictions": "GET list, GET /:id, PATCH /:id, DELETE /:id - CRUD + search",
            "/api/v1/db/training-data": "GET list, GET /:id, PATCH /:id, DELETE /:id - CRUD + search",
            "/api/v1/db/model-performance": "GET list, GET /:id, PATCH /:id, DELETE /:id - CRUD + search",
            "/api/v1/mlops/status": "GET - Production/Staging model versions (Phase 4)",
            "/api/v1/mlops/metrics": "GET - Prometheus metrics (Phase 4)",
            "/api/v1/mlops/rollback": "POST - Rollback Production to a version (Phase 4)",
            "/health": "GET - Check API health",
            "/model/info": "GET - Get model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "API is running"
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES.values())
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
