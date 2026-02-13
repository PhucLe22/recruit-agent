from fastapi import APIRouter
from app.api.v1.endpoints import users, health, ai_matching, resume, jobs

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(ai_matching.router, prefix="/ai", tags=["ai-matching"])
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
