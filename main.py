from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from router.process import router
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

# Include router
app.include_router(router)


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
