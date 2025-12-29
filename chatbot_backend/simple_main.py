"""
Simplified CV Project API - Job Matching System Only
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Optional

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")
if not MONGO_ATLAS_URI:
    raise ValueError("MONGO_ATLAS_URI not found in environment")

DB_NAME = "CVProject"

# ==================== DATABASE SETUP ====================
mongo_client = MongoClient(MONGO_ATLAS_URI)
db = mongo_client[DB_NAME]
users_collection = db["users"]
cvs_collection = db["cvs"]
jobs_collection = db["jobs"]

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="CV Project API - Simplified",
    description="Resume Management & Job Matching System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:5173'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HELPER FUNCTIONS ====================
def calculate_job_match(job: dict, skills: set, titles: list) -> int:
    """Calculate match percentage for a job with enhanced matching"""
    score = 0
    
    # Title match (30%)
    job_title = job.get('title', '').lower()
    title_match = any(t.lower() in job_title for t in titles)
    if title_match:
        score += 30
    
    # Enhanced Skills match (50%)
    all_job_skills = set()
    
    # Check multiple skill fields
    skill_fields = ['required_skills', 'skills', 'technique', 'preferred_skills']
    for field in skill_fields:
        if field in job and job[field]:
            if isinstance(job[field], list):
                all_job_skills.update(s.lower().strip() for s in job[field] if s)
            elif isinstance(job[field], str):
                all_job_skills.update(s.strip().lower() for s in job[field].split(',') if s.strip())
    
    # Also check description and requirements for skill mentions
    text_fields = [job.get('description', ''), job.get('requirements', '')]
    for text in text_fields:
        if text:
            text_lower = text.lower()
            # Find mentioned skills in text
            for skill in skills:
                if skill in text_lower:
                    all_job_skills.add(skill)
    
    if all_job_skills:
        matched = len(skills & all_job_skills)
        skill_ratio = len(matched) / max(len(all_job_skills), 1)
        score += skill_ratio * 50
    
    # Experience level bonus (15%)
    exp_levels = {'entry': 1, 'mid': 2, 'senior': 3, 'executive': 4}
    job_exp = job.get('experience_level', '').lower()
    if job_exp in exp_levels:
        score += 15  # Bonus for having experience level info
    
    # Industry/location bonus (5%)
    if job.get('industries') or job.get('location'):
        score += 5
    
    return min(100, int(score))

def _get_matched_skills(job: dict, resume_skills: set) -> list:
    """Extract which skills from resume matched the job"""
    matched = []
    
    # Check all job skill fields
    all_job_skills = set()
    skill_fields = ['required_skills', 'skills', 'technique', 'preferred_skills']
    
    for field in skill_fields:
        if field in job and job[field]:
            if isinstance(job[field], list):
                all_job_skills.update(s.lower().strip() for s in job[field] if s)
            elif isinstance(job[field], str):
                all_job_skills.update(s.strip().lower() for s in job[field].split(',') if s.strip())
    
    # Check text fields too
    text_fields = [job.get('description', ''), job.get('requirements', '')]
    for text in text_fields:
        if text:
            text_lower = text.lower()
            for skill in resume_skills:
                if skill in text_lower:
                    all_job_skills.add(skill)
    
    # Find intersection
    matched = list(resume_skills & all_job_skills)
    return matched

def extract_skills_from_text(resume_text: str) -> dict:
    """Simple skill extraction using keyword patterns"""
    # Common tech skills
    tech_skills = {
        'react', 'vue', 'angular', 'javascript', 'typescript', 'nodejs', 'node.js',
        'python', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'html', 'css', 'tailwind', 'bootstrap', 'sass', 'less',
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'sqlite',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'django', 'flask', 'express', 'spring boot', 'fastapi',
        'git', 'github', 'gitlab', 'ci/cd', 'jenkins', 'travis'
    }
    
    text_lower = resume_text.lower()
    found_skills = set()
    
    for skill in tech_skills:
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            found_skills.add(skill)
    
    # Extract job titles
    job_titles = []
    title_patterns = [
        r'\b(senior|junior|lead|principal)?\s*(software|web|frontend|backend|full[-]?stack)\s+(developer|engineer)\b',
        r'\b(senior|junior|lead|principal)?\s*(data|machine\slearning)\s+(scientist|engineer)\b',
        r'\b(devops|site\sreliability)\s+engineer\b',
        r'\b(product|project)\s+manager\b',
        r'\b(ui|ux)\s+designer\b'
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                title = ' '.join(filter(None, match))
            else:
                title = match
            job_titles.append(title.title())
    
    return {
        "skills": list(found_skills),
        "job_titles": job_titles
    }

# ==================== JOB ENDPOINTS ====================
@app.get("/api/jobs-suggestion/{username}")
async def jobs_suggestion(username: str):
    """Get AI-powered job recommendations based on resume"""
    try:
        # Get resume
        doc = cvs_collection.find_one({"username": username})
        if not doc or "processed_text" not in doc:
            raise HTTPException(404, "Resume not found")
        
        # Extract skills and titles from resume
        extracted_data = extract_skills_from_text(doc["processed_text"])
        skills = set(s.lower() for s in extracted_data.get("skills", []))
        titles = extracted_data.get("job_titles", [])
        
        # Enhanced job query - check more fields and be more inclusive
        query_conditions = []
        
        # 1. Skills matching in multiple fields
        if skills:
            skill_conditions = []
            for skill in skills:
                skill_conditions.extend([
                    {"required_skills": {"$regex": skill, "$options": "i"}},
                    {"skills": {"$regex": skill, "$options": "i"}},
                    {"technique": {"$regex": skill, "$options": "i"}},
                    {"description": {"$regex": skill, "$options": "i"}},
                    {"requirements": {"$regex": skill, "$options": "i"}}
                ])
            if skill_conditions:
                query_conditions.append({"$or": skill_conditions})
        
        # 2. Title matching
        if titles:
            title_conditions = []
            for title in titles:
                title_conditions.append({"title": {"$regex": title, "$options": "i"}})
            if title_conditions:
                query_conditions.append({"$or": title_conditions})
        
        # 3. Fallback - get recent jobs if no specific matches
        if not query_conditions:
            query_conditions.append({})  # Match all
        
        # Combine conditions with OR for broader matching
        final_query = {"$or": query_conditions} if len(query_conditions) > 1 else query_conditions[0]
        
        # Get more jobs for better matching
        all_jobs = list(jobs_collection.find(final_query, {
            "_id": 1, "title": 1, "companyName": 1, "company": 1,
            "required_skills": 1, "preferred_skills": 1, "skills": 1,
            "technique": 1, "description": 1, "requirements": 1,
            "slug": 1, "city": 1, "location": 1, "type": 1,
            "experience_level": 1, "industries": 1
        }).limit(100))  # Increased limit for better matching
        
        matched_jobs = []
        for job in all_jobs:
            match_pct = calculate_job_match(job, skills, titles)
            if match_pct >= 15:  # Lowered threshold for more matches
                matched_jobs.append({
                    "id": str(job["_id"]),
                    "title": job.get("title", "No Title"),
                    "company": job.get("companyName", job.get("company", "N/A")),
                    "slug": job.get("slug", f"job-{job['_id']}"),
                    "location": job.get("city", job.get("location", "Location")),
                    "type": job.get("type", "Full-time"),
                    "match_percentage": match_pct,
                    "relevance": "high" if match_pct >= 60 else "medium" if match_pct >= 35 else "low",
                    "matched_skills": _get_matched_skills(job, skills)
                })
        
        # Sort by match percentage
        matched_jobs.sort(key=lambda x: x["match_percentage"], reverse=True)
        
        # If still no matches, get some recent jobs as fallback
        if not matched_jobs:
            recent_jobs = list(jobs_collection.find({}, {
                "_id": 1, "title": 1, "companyName": 1, "company": 1,
                "slug": 1, "city": 1, "location": 1, "type": 1
            }).sort("_id", -1).limit(10))
            
            for job in recent_jobs:
                matched_jobs.append({
                    "id": str(job["_id"]),
                    "title": job.get("title", "No Title"),
                    "company": job.get("companyName", job.get("company", "N/A")),
                    "slug": job.get("slug", f"job-{job['_id']}"),
                    "location": job.get("city", job.get("location", "Location")),
                    "type": job.get("type", "Full-time"),
                    "match_percentage": 20,  # Default low match for fallback
                    "relevance": "low",
                    "matched_skills": []
                })
        
        return {
            "matching_jobs": matched_jobs[:30],  # Increased result count
            "total_matches": len(matched_jobs),
            "extracted_data": {
                "skills": list(skills),
                "job_titles": titles
            },
            "search_query": final_query  # Debug info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job suggestion error: {e}")
        return JSONResponse(200, {
            "matching_jobs": [],
            "total_matches": 0,
            "error": str(e)
        })

# ==================== HEALTH CHECK ====================
@app.get("/")
def root():
    """API root"""
    return {
        "name": "CV Project API - Simplified",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
