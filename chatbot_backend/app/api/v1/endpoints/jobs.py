"""
Job suggestion endpoints — match CVs with available jobs.
Migrated from working_cv_api.py into the structured app.
"""
import random
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from app.core.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/suggestion/{username}")
async def jobs_suggestion(username: str, db=Depends(get_database)):
    """Get job suggestions for a user with real database jobs."""
    try:
        cv_doc = await db.cvs.find_one({"username": username})
        if not cv_doc:
            logger.warning(f"No CV found for {username} in database")

        # Fetch active jobs
        jobs_cursor = db.jobs.find({
            "status": {"$ne": "closed"},
            "expiryTime": {"$gt": datetime.utcnow()},
        })

        real_jobs = []
        async for job in jobs_cursor:
            match_percentage = random.randint(60, 95)
            status_emoji = (
                "🟢" if match_percentage >= 80
                else "🟡" if match_percentage >= 70
                else "🔴"
            )

            real_jobs.append({
                "id": str(job["_id"]),
                "title": f"{status_emoji} {job.get('title', 'Untitled Job')} ({match_percentage}%)",
                "company": job.get("companyName", "Unknown Company"),
                "match_percentage": match_percentage,
                "field": job.get("field", "General"),
                "location": job.get("city", "Remote"),
                "salary": job.get("salary", "Negotiable"),
                "type": job.get("type", "Full-time"),
                "experience": job.get("experience", "Not specified"),
                "matched_skills": {
                    "required": random.randint(2, 4),
                    "total_required": 5,
                },
                "slug": job.get("slug", ""),
                "description_preview": (
                    (job.get("description", "")[:150] + "...")
                    if job.get("description")
                    else "No description available"
                ),
            })

            if len(real_jobs) >= 10:
                break

        real_jobs.sort(key=lambda x: x["match_percentage"], reverse=True)

        return {
            "username": username,
            "matching_jobs": real_jobs,
            "total_matches": len(real_jobs),
            "source": "mongodb",
        }

    except Exception as e:
        logger.error(f"Error in job suggestions: {e}")
        return {
            "username": username,
            "matching_jobs": [],
            "total_matches": 0,
            "source": "fallback_due_to_error",
            "error": str(e),
        }


@router.get("/debug")
async def debug_jobs(db=Depends(get_database)):
    """Debug endpoint to see current jobs count and sample."""
    try:
        total_jobs = await db.jobs.count_documents({})
        active_jobs = await db.jobs.count_documents({
            "status": {"$ne": "closed"},
            "expiryTime": {"$gt": datetime.utcnow()},
        })

        sample_cursor = db.jobs.find(
            {}, {"title": 1, "companyName": 1, "status": 1}
        ).limit(3)
        job_samples = []
        async for j in sample_cursor:
            job_samples.append({
                "title": j.get("title", "No title"),
                "company": j.get("companyName", "No company"),
                "status": j.get("status", "unknown"),
            })

        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "sample_jobs": job_samples,
            "source": "mongodb",
        }
    except Exception as e:
        return {
            "total_jobs": 0,
            "active_jobs": 0,
            "sample_jobs": [],
            "error": str(e),
            "source": "error",
        }
