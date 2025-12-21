"""
CV Project API - Resume Management & Job Matching System
"""

import os
import re
import pickle
import logging
from datetime import datetime
from typing import List, Optional

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from services.ingestion.pipeline import pipeline
from services.cv_refinement.improvement_suggestion import suggest_resume_improvements
from services.cv_refinement.jobs_suggestion import suggest_jobs
from services.linkedin_webscraping.webscraping import retrieve_linkedin_jobs
import config

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")
if not MONGO_ATLAS_URI:
    raise ValueError("MONGO_ATLAS_URI not found in environment")

DB_NAME = "CVProject"
UPLOAD_DIR = "uploads"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== DATABASE SETUP ====================
mongo_client = MongoClient(MONGO_ATLAS_URI)
db = mongo_client[DB_NAME]
users_collection = db["users"]
cvs_collection = db["cvs"]
jobs_collection = db["jobs"]

# Create indexes
try:
    users_collection.create_index("username", unique=True)
    cvs_collection.create_index("username", unique=True)
    jobs_collection.create_index("unique_id", unique=True)
except Exception:
    pass

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="CV Project API",
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

# ==================== MODELS ====================
class GoogleMeetRequest(BaseModel):
    summary: str
    start_time: str  # ISO format
    end_time: str
    timezone: str = "Asia/Ho_Chi_Minh"
    attendees: List[str] = []
    description: str = ""

# ==================== HELPER FUNCTIONS ====================
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file extension and size"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file format. Allowed: {SUPPORTED_EXTENSIONS}")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max: 5MB, Yours: {file_size/1024/1024:.2f}MB")
    if file_size == 0:
        raise HTTPException(400, "File is empty")

def get_google_credentials():
    """Get or refresh Google OAuth2 credentials"""
    creds = None
    token_file = 'token.pickle'
    
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            os.remove(token_file) if os.path.exists(token_file) else None
            return None
    
    return creds

def calculate_job_match(job: dict, skills: set, titles: list) -> int:
    """Calculate match percentage for a job"""
    score = 0
    
    # Title match (40%)
    job_title = job.get('title', '').lower()
    if any(t.lower() in job_title for t in titles):
        score += 40
    
    # Skills match (50%)
    required_skills = set(s.lower() for s in job.get('required_skills', []))
    if required_skills:
        matched = len(skills & required_skills)
        score += (matched / len(required_skills)) * 50
    
    # Base score (10%)
    score += 10
    
    return min(100, int(score))

# ==================== USER ENDPOINTS ====================
@app.post("/create_user")
def create_user(username: str = Form(...)):
    """Create a new user"""
    try:
        users_collection.insert_one({
            "username": username,
            "created_at": datetime.utcnow()
        })
        return {"username": username, "created": True, "message": "User created"}
    except DuplicateKeyError:
        return {"username": username, "created": False, "message": "User exists"}
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(500, "User creation failed")

@app.get("/users")
def get_users():
    """Get all users"""
    try:
        users = list(users_collection.find({}, {"_id": 0}))
        return {"users": users, "count": len(users)}
    except Exception as e:
        logger.error(f"Get users error: {e}")
        raise HTTPException(500, "Failed to fetch users")

# ==================== RESUME ENDPOINTS ====================
@app.post("/upload_resume")
async def upload_resume(
    username: str = Form(...),
    file: UploadFile = File(...),
    model_name: str = config.MODEL_NAME
):
    """Upload and process resume"""
    validate_file(file)
    
    # Create user if not exists
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({
            "username": username,
            "email": f"{username}@cvproject.com",
            "created_at": datetime.utcnow()
        })
    
    # Save file
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{username}__{timestamp}__{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        processed_text, parsed_output = pipeline(file_path)
        
        result = cvs_collection.update_one(
            {"username": username},
            {"$set": {
                "username": username,
                "uploaded_at": datetime.utcnow(),
                "processed_text": processed_text,
                "parsed_output": parsed_output
            }},
            upsert=True
        )
        
        return {
            "username": username,
            "saved": True,
            "inserted_id": str(result.upserted_id) if result.upserted_id else None,
            "message": "Resume uploaded successfully",
            "file_path": file_path,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        os.remove(file_path) if os.path.exists(file_path) else None
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/resume/{username}")
def get_resume(username: str):
    """Get user's resume"""
    doc = cvs_collection.find_one({"username": username}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Resume not found")
    return doc

@app.delete("/resume/{username}")
def delete_resume(username: str):
    """Delete user's resume"""
    result = cvs_collection.delete_one({"username": username})
    if result.deleted_count == 0:
        raise HTTPException(404, "Resume not found")
    return {"deleted": True, "username": username}

@app.post("/resume/{username}/suggest_improvements")
def resume_improvements(username: str, model_name: str = config.MODEL_NAME):
    """Get AI-powered resume improvement suggestions"""
    doc = cvs_collection.find_one({"username": username})
    if not doc or "processed_text" not in doc:
        raise HTTPException(404, "Resume not found")
    
    try:
        improvements = suggest_resume_improvements(doc["processed_text"], model_name)
        return {"username": username, "model": model_name, "improvements": improvements}
    except Exception as e:
        logger.error(f"Improvement suggestion error: {e}")
        raise HTTPException(500, f"Failed to generate suggestions: {str(e)}")

# ==================== JOB ENDPOINTS ====================
@app.get("/users/{username}/jobs")
def retrieve_jobs(username: str, headless: bool = True):
    """Scrape LinkedIn jobs for user"""
    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(404, "User not found")
    
    keywords = user.get("keywords", [])
    if not keywords:
        raise HTTPException(400, "No keywords stored")
    
    params_list = [{"keywords": kw, "location": "Ho Chi Minh City"} for kw in keywords]
    
    try:
        result = retrieve_linkedin_jobs(headless=headless, params_list=params_list)
        
        jobs_to_insert = []
        for job in result["jobs"]:
            job["unique_id"] = f"{username}_{job['job_id']}"
            job["user"] = username
            if not jobs_collection.find_one({"unique_id": job["unique_id"]}):
                jobs_to_insert.append(job)
        
        if jobs_to_insert:
            jobs_collection.insert_many(jobs_to_insert)
        
        user_jobs = list(jobs_collection.find({"user": username}, {"_id": 0}))
        return {"count": len(user_jobs), "jobs": user_jobs}
        
    except Exception as e:
        logger.error(f"Job retrieval error: {e}")
        raise HTTPException(500, f"Failed to retrieve jobs: {str(e)}")

@app.get("/api/jobs-suggestion/{username}")
async def jobs_suggestion(username: str):
    """Get AI-powered job recommendations based on resume"""
    try:
        # Get resume
        doc = cvs_collection.find_one({"username": username})
        if not doc or "processed_text" not in doc:
            raise HTTPException(404, "Resume not found")
        
        # Extract skills and titles from resume
        job_suggestions = suggest_jobs(doc["processed_text"], model_name="qwen2.5:3b", limit=10)
        skills = set(s.lower() for s in job_suggestions.get("extracted_data", {}).get("skills", []))
        titles = job_suggestions.get("extracted_data", {}).get("job_titles", [])
        
        # Get and match jobs
        all_jobs = list(jobs_collection.find({}, {
            "_id": 1, "title": 1, "companyName": 1, "company": 1,
            "required_skills": 1, "preferred_skills": 1
        }).limit(50))
        
        matched_jobs = []
        for job in all_jobs:
            match_pct = calculate_job_match(job, skills, titles)
            if match_pct >= 30:  # Only include decent matches
                matched_jobs.append({
                    "id": str(job["_id"]),
                    "title": job.get("title", "No Title"),
                    "company": job.get("companyName", job.get("company", "N/A")),
                    "match_percentage": match_pct,
                    "relevance": "high" if match_pct >= 70 else "medium" if match_pct >= 50 else "low"
                })
        
        # Sort by match percentage
        matched_jobs.sort(key=lambda x: x["match_percentage"], reverse=True)
        
        return {
            "matching_jobs": matched_jobs[:20],
            "total_matches": len(matched_jobs),
            "extracted_data": {
                "skills": list(skills),
                "job_titles": titles
            }
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

# ==================== AI APPLICANT MATCHING ENDPOINTS ====================

@app.get("/api/applicant-matching/{job_id}")
async def get_applicant_matching(job_id: str):
    """Get AI-powered applicant matching for a specific job"""
    try:
        import requests
        
        # Call the Node.js AI matching service
        node_service_url = "http://localhost:3000/business/api/job"
        
        response = requests.get(f"{node_service_url}/{job_id}/matching-applicants", 
                               timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"Service returned status {response.status_code}",
                "applicants": []
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"AI matching service error: {e}")
        return {
            "success": False,
            "message": "AI matching service unavailable",
            "applicants": []
        }
    except Exception as e:
        logger.error(f"Applicant matching error: {e}")
        raise HTTPException(500, f"Failed to get applicant matching: {str(e)}")

@app.get("/api/applicant-matching/all/{business_id}")
async def get_all_applicant_matching(business_id: str):
    """Get AI-powered applicant matching for all jobs of a business"""
    try:
        import requests
        
        # Call the Node.js AI matching service
        node_service_url = "http://localhost:3000/business/api"
        
        response = requests.get(f"{node_service_url}/matching-applicants/all", 
                               timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"Service returned status {response.status_code}",
                "recommendations": {}
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"AI matching service error: {e}")
        return {
            "success": False,
            "message": "AI matching service unavailable",
            "recommendations": {}
        }
    except Exception as e:
        logger.error(f"Applicant matching error: {e}")
        raise HTTPException(500, f"Failed to get applicant matching: {str(e)}")

@app.get("/api/applicant-profile/{user_id}")
async def get_applicant_profile(user_id: str, job_id: str = None):
    """Get detailed applicant profile with AI matching insights"""
    try:
        import requests
        
        # Call the Node.js AI matching service with API key for authentication
        node_service_url = "http://localhost:3000/business/api"
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "ai-service-internal-key"  # Simple API key for internal service communication
        }
        
        url = f"{node_service_url}/applicant/{user_id}/profile"
        if job_id:
            url += f"?jobId={job_id}"
            
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"Service returned status {response.status_code}: {response.text}",
                "data": None
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"AI profile service error: {e}")
        return {
            "success": False,
            "message": "AI profile service unavailable",
            "data": None
        }
    except Exception as e:
        logger.error(f"Applicant profile error: {e}")
        raise HTTPException(500, f"Failed to get applicant profile: {str(e)}")

# ==================== GOOGLE MEET ENDPOINTS ====================
SCOPES = ['https://www.googleapis.com/auth/calendar']

@app.get("/api/auth/google")
async def auth_google():
    """Initiate Google OAuth2 flow"""
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json',
            SCOPES,
            redirect_uri=os.getenv('GOOGLE_REDIRECT_URI')
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return {"auth_url": auth_url}
    except Exception as e:
        logger.error(f"Auth URL error: {e}")
        raise HTTPException(500, str(e))

@app.get("/auth/google/callback")
async def auth_callback(code: str):
    """Handle Google OAuth2 callback"""
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES,
            redirect_uri=os.getenv('GOOGLE_REDIRECT_URI')
        )
        flow.fetch_token(code=code)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(flow.credentials, token)
            
        return {"status": "success", "message": "Authenticated with Google"}
    except Exception as e:
        logger.error(f"Auth callback error: {e}")
        raise HTTPException(400, str(e))

@app.post("/api/create-meet")
async def create_google_meet(meet_request: GoogleMeetRequest):
    """Create a Google Meet meeting"""
    try:
        creds = get_google_credentials()
        if not creds:
            return {
                "error": "authentication_required",
                "message": "Visit /api/auth/google to authenticate"
            }
        
        service = build('calendar', 'v3', credentials=creds)
        
        event = {
            'summary': meet_request.summary,
            'description': meet_request.description,
            'start': {'dateTime': meet_request.start_time, 'timeZone': meet_request.timezone},
            'end': {'dateTime': meet_request.end_time, 'timeZone': meet_request.timezone},
            'conferenceData': {
                'createRequest': {
                    'requestId': f"meet-{int(datetime.utcnow().timestamp())}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            },
            'conferenceDataVersion': 1
        }
        
        if meet_request.attendees:
            event['attendees'] = [{'email': email} for email in meet_request.attendees]
        
        event = service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1
        ).execute()
        
        return {
            "meet_link": event.get('hangoutLink', ''),
            "event_id": event['id'],
            "html_link": event.get('htmlLink', '')
        }
        
    except Exception as e:
        logger.error(f"Meet creation error: {e}")
        if "invalid_grant" in str(e) or "token" in str(e).lower():
            os.remove('token.pickle') if os.path.exists('token.pickle') else None
            return {
                "error": "authentication_required",
                "message": "Re-authenticate with Google"
            }
        raise HTTPException(500, str(e))

# ==================== HEALTH CHECK ====================
@app.get("/")
def root():
    """API root"""
    return {
        "name": "CV Project API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)