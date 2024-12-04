# app/main.py
from fastapi import FastAPI
from app.api.routers import no_show_prediction, brazil_prediction
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical No-Show Prediction Service",
    description="API for predicting patient appointment no-shows for California and Brazil datasets",
    version="1.0.0"
)

# Register California router
app.include_router(
    no_show_prediction.router,
    prefix="/api/v1/california",
    tags=["California No-Show Prediction"]
)

# Register Brazil router
app.include_router(
    brazil_prediction.router,
    prefix="/api/v1/brazil",
    tags=["Brazil No-Show Prediction"]
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "california_model": {
            "loaded": hasattr(no_show_prediction, 'model'),
            "config_loaded": hasattr(no_show_prediction, 'config')
        },
        "brazil_model": {
            "loaded": hasattr(brazil_prediction, 'model'),
            "config_loaded": hasattr(brazil_prediction, 'config')
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
