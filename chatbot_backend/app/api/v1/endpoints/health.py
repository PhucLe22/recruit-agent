from datetime import datetime
from fastapi import APIRouter
from app.models.schemas import HealthCheck

router = APIRouter()


@router.get("/", response_model=HealthCheck)
def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="ok",
        time=datetime.utcnow()
    )


@router.get("/root")
def root():
    """API root"""
    return {
        "name": "CV Project API",
        "version": "1.0.0",
        "docs": "/docs"
    }
