"""
Resume endpoints — upload, retrieve, and analyze CVs.
Migrated from working_cv_api.py into the structured app.
"""
import os
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from app.core.config import settings
from app.core.database import get_database
from app.services.cv_refinement.cv_analysis import analyze_cv_content

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"}


@router.post("/upload")
async def upload_resume(
    username: str = Form(...),
    file: UploadFile = File(...),
    db=Depends(get_database),
):
    """Upload and process a resume."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file format: {file_ext}"
        )

    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="The file is empty")

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size // (1024*1024)}MB",
        )

    try:
        file_content = await file.read()
        processed_text = f"Processed {file.filename} ({len(file_content)} bytes)"

        doc = {
            "username": username,
            "uploaded_at": datetime.utcnow(),
            "filename": file.filename,
            "processed_text": processed_text,
            "file_size": file_size,
            "file_type": file_ext,
        }

        await db.cvs.update_one(
            {"username": username}, {"$set": doc}, upsert=True
        )

        return {
            "username": username,
            "saved": True,
            "message": f"Resume '{file.filename}' uploaded successfully.",
            "filename": file.filename,
            "size": file_size,
        }

    except Exception as e:
        logger.error(f"Error uploading resume for {username}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading resume: {str(e)}"
        )


@router.get("/{username}")
async def get_resume(username: str, db=Depends(get_database)):
    """Get stored resume for a user."""
    doc = await db.cvs.find_one({"username": username}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found for that username.")
    return doc


@router.post("/{username}/suggest_improvements")
async def suggest_improvements(username: str, db=Depends(get_database)):
    """Suggest detailed improvements for a resume based on actual CV content."""
    doc = await db.cvs.find_one({"username": username})
    if not doc:
        raise HTTPException(
            status_code=404, detail="No resume found for that username."
        )

    cv_content = doc.get("processed_text", "")
    filename = doc.get("filename", "CV")
    file_size = doc.get("file_size", 0)
    file_type = doc.get("file_type", "")
    uploaded_at = doc.get("uploaded_at", "")

    try:
        analysis_result = analyze_cv_content(cv_content, filename, file_size, file_type)
    except Exception as e:
        logger.error(f"Error analyzing resume for {username}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing resume: {str(e)}"
        )

    return {
        "username": username,
        "analysis": analysis_result,
        "cv_metadata": {
            "filename": filename,
            "file_size": file_size,
            "file_type": file_type,
            "uploaded_at": uploaded_at.isoformat() if hasattr(uploaded_at, "isoformat") else uploaded_at,
        },
    }
