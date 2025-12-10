# main.py

import os
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Optional

# Load environment variables from .env file
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root directory
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import uvicorn
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from services.ingestion.pipeline import pipeline
from services.cv_refinement.improvement_suggestion import suggest_resume_improvements
from services.linkedin_webscraping.webscraping import retrieve_linkedin_jobs
import config
from config import CreateUserResp, UploadResp, UserResp, GetUsersResp
from services.cv_refinement.jobs_suggestion import suggest_jobs, extract_job_titles_from_resume, extract_skills_from_text

def get_default_kills_for_title(title: str) -> list:
    """Get default skills based on job title"""
    title_lower = title.lower()
    
    # Map of job title keywords to default skills with more specific and relevant skills
    skills_map = {
        'backend': [
            'python', 'java', 'node.js', 'sql', 'rest api', 'git', 'docker', 'aws',
            'mongodb', 'postgresql', 'mysql', 'graphql', 'microservices', 'api development'
        ],
        'fullstack': [
            'javascript', 'react', 'node.js', 'python', 'sql', 'html', 'css', 'rest api',
            'typescript', 'redux', 'next.js', 'express', 'mongodb', 'postgresql', 'git', 'docker'
        ],
        'frontend': [
            'javascript', 'react', 'vue', 'angular', 'html', 'css', 'typescript', 'responsive design',
            'redux', 'sass', 'next.js', 'webpack', 'babel', 'jest', 'testing'
        ],
        'devops': [
            'docker', 'kubernetes', 'aws', 'ci/cd', 'terraform', 'linux', 'bash', 'python',
            'jenkins', 'github actions', 'ansible', 'prometheus', 'grafana', 'nginx', 'cloud'
        ],
        'data scientist': [
            'python', 'machine learning', 'pandas', 'numpy', 'sql', 'statistics', 'tensorflow', 'pytorch',
            'data analysis', 'data visualization', 'scikit-learn', 'deep learning', 'jupyter', 'matplotlib', 'seaborn'
        ],
        'mobile': [
            'swift', 'kotlin', 'react native', 'flutter', 'mobile ui/ux', 'rest api', 'firebase',
            'ios development', 'android development', 'mobile app development', 'redux', 'typescript', 'graphql'
        ],
        'ai': [
            'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision',
            'neural networks', 'data science', 'natural language processing', 'opencv', 'scikit-learn', 'keras'
        ],
        'cloud': [
            'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform', 'ci/cd',
            'serverless', 'lambda', 'cloudformation', 'ansible', 'jenkins', 'github actions', 'devops'
        ],
        'lập trình viên': [
            'javascript', 'python', 'java', 'sql', 'git', 'oop', 'algorithms', 'data structures',
            'html', 'css', 'react', 'node.js', 'mongodb', 'rest api', 'typescript', 'docker'
        ],
        'tester': [
            'testing', 'automation', 'selenium', 'junit', 'testng', 'api testing', 'manual testing',
            'jest', 'cypress', 'postman', 'jira', 'test automation', 'qa', 'quality assurance'
        ],
        'digital marketing': [
            'seo', 'sem', 'social media marketing', 'content marketing', 'google analytics',
            'email marketing', 'ppc', 'digital advertising', 'marketing strategy', 'seo'
        ],
        'hr': [
            'recruitment', 'talent acquisition', 'employee relations', 'hr policies', 'performance management',
            'training and development', 'compensation and benefits', 'hr management', 'onboarding', 'labor law'
        ],
        'sales': [
            'customer relationship management', 'sales strategy', 'business development', 'account management',
            'negotiation', 'market research', 'sales presentations', 'b2b sales', 'sales forecasting', 'crm'
        ]
    }
    
    # Find matching skills based on title
    default_skills = []
    for keyword, skill_list in skills_map.items():
        if keyword in title_lower:
            default_skills.extend(skill_list)
    
    # If no specific match, return some general skills
    if not default_skills:
        default_skills = ['problem solving', 'teamwork', 'communication', 'git', 'agile', 'debugging']
    
    return list(set(default_skills))  # Remove duplicates

import logging

# ---------- LOGGER SETUP ----------
logging.basicConfig(
    level=logging.INFO,  # can be DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------- CONFIG ----------
logger.info(os.getcwd())
load_dotenv()

MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")
if not MONGO_ATLAS_URI:
    raise ValueError("MONGO_ATLAS_URI not found in environment variables. Please set it in your .env file.")

DB_NAME = "CVProject"
USERS_COLLECTION = "users"
CVS_COLLECTION = "cvs"
JOBS_COLLECTION = "jobs"

UPLOAD_DIR = "uploads"
OCR_DIR = "ocr_text"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

# ---------- APP ----------
app = FastAPI(title="Resume Ingestion API (Simple User)")

origins = [
    "http://localhost:5173",
]

# allow your frontend domain(s) here
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:5173'],  # change to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB CLIENT ----------
mongo_client = MongoClient(MONGO_ATLAS_URI)
db = mongo_client[DB_NAME]
users_collection = db[USERS_COLLECTION]
cvs_collection = db[CVS_COLLECTION]
jobs_collection = db[JOBS_COLLECTION]

# Optional: ensure an index on username (unique for users)
try:
    users_collection.create_index("username", unique=True)
    cvs_collection.create_index("username", unique=True)  # ensures one resume per username
    jobs_collection.create_index("unique_id", unique=True)
except Exception:
    # index creation might error if running multiple times; ignore for simple setup
    pass

# ---------- Endpoints ----------

@app.post("/create_user", response_model=CreateUserResp)
def create_user(username: str = Form(...)):
    logger.info(f"Received request to create user: {username}")
    now = datetime.utcnow()
    user_doc = {"username": username, "created_at": now}

    try:
        users_collection.insert_one(user_doc)
        logger.info(f"User '{username}' created successfully.")
        return CreateUserResp(
            username=username,
            created=True,
            message="User created successfully."
        )

    except DuplicateKeyError:
        logger.warning(f"User '{username}' already exists in the database.")
        return CreateUserResp(
            username=username,
            created=False,
            message="User already exists."
        )

    except Exception as e:
        logger.error(f"Unexpected error creating user '{username}': {e}")
        return CreateUserResp(
            username=username,
            created=False,
            message="An unexpected error occurred."
        )

@app.get("/users", response_model=GetUsersResp)
def get_users():
    """
    Retrieve all users from the database.
    Returns a list of users with their details.
    """
    logger.info("Received request to get all users")

    try:
        # Fetch all users from the database
        users = list(users_collection.find({}, {"_id": 0}))

        # Convert users to response format
        user_responses = []
        for user in users:
            user_response = UserResp(
                username=user.get("username", ""),
                email=user.get("email"),
                created_at=user.get("created_at").isoformat() if user.get("created_at") else None,
                keywords=user.get("keywords", [])
            )
            user_responses.append(user_response)

        logger.info(f"Successfully retrieved {len(user_responses)} users")
        return GetUsersResp(
            users=user_responses,
            count=len(user_responses),
            message="Users retrieved successfully."
        )

    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        return GetUsersResp(
            users=[],
            count=0,
            message=f"Error retrieving users: {str(e)}"
        )

# List of supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}

@app.post("/upload_resume", response_model=UploadResp)
async def upload_resume(username: str = Form(...), file: UploadFile = File(...), model_name: str = config.MODEL_NAME):
    logger.info(f"Uploading resume for user: {username}, file: {file.filename}")

    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Check file size (max 5MB)
    max_size = 5 * 1024 * 1024  # 5MB
    file.file.seek(0, 2)  # Move to end of file
    file_size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 5MB. Your file is {file_size/1024/1024:.2f}MB"
        )
    elif file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="The file is empty"
        )

    # Create user if not exists
    if users_collection.find_one({"username": username}) is None:
        logger.info(f"User '{username}' not found. Creating automatically.")
        try:
            users_collection.insert_one({
                "username": username,
                "email": f"{username}@cvproject.com",  # Add a unique email
                "created_at": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Error creating user account")

    # Save the uploaded file
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{username}__{timestamp}__{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        logger.info(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Unable to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Unable to save uploaded file: {e}")

    try:
        processed_text, parsed_output = pipeline(file_path)
        logger.info(f"Pipeline processed file for user '{username}' successfully.")
    except Exception as e:
        logger.error(f"Pipeline error for user '{username}': {e}")
        try:
            os.remove(file_path)
        except Exception:
            logger.warning(f"Failed to remove file after pipeline error: {file_path}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    doc = {
        "username": username,
        "uploaded_at": datetime.utcnow(),
        "processed_text": processed_text,
        "parsed_output": parsed_output,
    }

    try:
        result = cvs_collection.update_one({"username": username}, {"$set": doc}, upsert=True)
        inserted_id = str(result.upserted_id) if result.upserted_id else None
        logger.info(f"Resume stored for '{username}', replaced existing: {result.matched_count > 0}")
        # return UploadResp(username=username, saved=True, inserted_id=inserted_id,
        #                   message="Resume stored (replaced if existed).")\\\

        res = {
            "username": username,
            "saved": True,
            "inserted_id": inserted_id,
            "message": "Resume stored (replaced if existed)."
        }
        logger.info(f"Curent Data: {res}")
        return res

    except Exception as e:
        logger.error(f"DB insert/update failed for user '{username}': {e}")
        raise HTTPException(status_code=500, detail=f"DB insert/update failed: {e}")


@app.get("/resume/{username}")
def get_resume(username: str):
    """Fetch the stored resume document for a username."""
    doc = cvs_collection.find_one({"username": username}, {"_id": 0})  # hide Mongo _id in response
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found for that username.")
    return doc

@app.delete("/resume/{username}")
def delete_resume(username: str):
    """Delete stored resume for testing / cleanup (optional)."""
    res = cvs_collection.delete_one({"username": username})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No resume found to delete.")
    return {"deleted": True, "username": username}

@app.post("/resume/{username}/suggest_improvements")
def resume_improvements(username: str, model_name: str = config.MODEL_NAME):
    """
    Suggest improvements for a user's resume.
    Runs the LLM pipeline on the resume text already stored in DB.
    """
    # fetch stored resume
    doc = cvs_collection.find_one({"username": username})
    if not doc or "processed_text" not in doc:
        raise HTTPException(status_code=404, detail="No resume found for that username.")

    resume_text = doc["processed_text"]

    try:
        improvements = suggest_resume_improvements(resume_text, model_name=model_name)
    except Exception as e:
        logger.error(f"Error suggesting improvements for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Improvement suggestion failed: {e}")

    return {
        "username": username,
        "model": model_name,
        "improvements": improvements
    }


@app.get("/users/{username}/jobs")
def retrieve_jobs(username: str, headless: bool = True):
    """
    Retrieve LinkedIn jobs for a specific user based on their stored keywords,
    store them in 'jobs' collection, and return them.
    """
    # 1. Get user
    user = users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")

    # 2. Build params list
    keywords = user.get("keywords", [])
    if not keywords:
        raise HTTPException(status_code=400, detail="No keywords stored for this user.")

    params_list = [{"keywords": kw, "location": "Ho Chi Minh City"} for kw in keywords]

    # 3. Scrape jobs
    try:
        result = retrieve_linkedin_jobs(headless=headless, params_list=params_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping jobs: {str(e)}")

    jobs_to_insert = []
    for job in result["jobs"]:
        # Ensure unique_id includes username
        job["unique_id"] = f"{username}_{job['job_id']}"
        job["user"] = username

        # Check if already exists
        if not jobs_collection.find_one({"unique_id": job["unique_id"]}):
            jobs_to_insert.append(job)

    # 4. Insert new jobs into DB
    if jobs_to_insert:
        jobs_collection.insert_many(jobs_to_insert)

    # 5. Return all jobs for this user
    user_jobs = list(jobs_collection.find({"user": username}, {"_id": 0}))
    return {"count": len(user_jobs), "jobs": user_jobs}

# ---------- Google Meet Endpoints ----------
class GoogleMeetRequest(BaseModel):
    summary: str
    start_time: str  # ISO format: "2025-10-10T14:00:00"
    end_time: str    # ISO format: "2025-10-10T15:00:00"
    timezone: str = "Asia/Ho_Chi_Minh"
    attendees: List[str] = []
    description: str = ""

# Google OAuth2 Configuration
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_google_credentials():
    """Get or refresh Google OAuth2 credentials."""
    creds = None
    token_file = 'token.pickle'
    
    # Load existing credentials if they exist
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing token: {e}")
                if os.path.exists(token_file):
                    os.remove(token_file)
                return None
        else:
            return None  # Need to authenticate
            
        # Save the refreshed credentials
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

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
        logger.error(f"Error generating auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/google/callback")
async def auth_callback(code: str):
    """Handle Google OAuth2 callback"""
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json',
            SCOPES,
            redirect_uri=os.getenv('GOOGLE_REDIRECT_URI')
        )
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
        return {"status": "success", "message": "Successfully authenticated with Google"}
        
    except Exception as e:
        logger.error(f"Error in auth callback: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/create-meet")
async def create_google_meet(meet_request: GoogleMeetRequest):
    """
    Create a Google Meet meeting.
    
    Requires a Google OAuth2 token (obtained via the OAuth2 flow).
    You'll need to set up a project in Google Cloud Console and download the credentials.json file.
    """
    try:
        creds = get_google_credentials()
        if not creds:
            return {
                "error": "authentication_required",
                "message": "Please authenticate with Google first by visiting /api/auth/google"
            }
            
        service = build('calendar', 'v3', credentials=creds)
        
        event = {
            'summary': meet_request.summary,
            'description': meet_request.description,
            'start': {
                'dateTime': meet_request.start_time,
                'timeZone': meet_request.timezone,
            },
            'end': {
                'dateTime': meet_request.end_time,
                'timeZone': meet_request.timezone,
            },
            'conferenceData': {
                'createRequest': {
                    'requestId': f"meet-{int(time.time())}",
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
            "meet_link": event.get('hangoutLink', event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', '')),
            "event_id": event['id'],
            "html_link": event.get('htmlLink', '')
        }
        
    except Exception as e:
        logger.error(f"Error creating meeting: {e}")
        if "invalid_grant" in str(e) or "token" in str(e).lower():
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
            return {
                "error": "authentication_required",
                "message": "Authentication expired. Please re-authenticate with Google."
            }
        raise HTTPException(status_code=500, detail=str(e))

def get_database():
    """Get database connection"""
    mongo_uri = os.getenv("MONGO_ATLAS_URI")
    if not mongo_uri:
        raise ValueError("MONGO_ATLAS_URI environment variable not set")
    client = MongoClient(mongo_uri)
    return client["CVProject"]

@app.get("/api/jobs-suggestion/{username}")
async def jobs_suggestion(username: str):
    try:
        # Initialize database connection
        try:
            db = get_database()
            users_collection = db["users"]
            cvs_collection = db["cvs"]
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not connect to the database")
            
        # Get user (if user verification is required)
        try:
            user = users_collection.find_one({"username": username})
            if not user:
                raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
        except Exception as e:
            logger.warning(f"User check skipped/error: {str(e)}")
            # Continue even if user check fails
        
        # Initialize result with empty lists
        result = {
            "matching_jobs": [],
            "total_matches": 0,
            "message": ""
        }
        
        # Initialize variables
        job_titles = []
        skills = set()
        
        try:
            doc = cvs_collection.find_one({"username": username})
            if doc and "processed_text" in doc:
                resume_text = doc["processed_text"]
                if resume_text and len(resume_text.strip()) > 0:
                    # First extract job titles and skills from resume
                    job_titles = extract_job_titles_from_resume(resume_text)
                    logger.info(f"Extracted job titles from CV: {job_titles}")
                    
                    # Get job suggestions and extract skills
                    job_suggestions = suggest_jobs(resume_text)
                    
                    # Extract skills from the result
                    if isinstance(job_suggestions, dict):
                        if "skills" in job_suggestions:
                            skills = set(skill["name"].lower() for skill in job_suggestions["skills"] if skill and isinstance(skill, dict) and skill.get("name"))
                        
                        # If no skills found, try to extract from the resume text
                        if not skills and "resume_text" in job_suggestions:
                            from services.cv_refinement.jobs_suggestion import extract_skills_from_text
                            skills = set(extract_skills_from_text(job_suggestions["resume_text"]))
                    
                    logger.info(f"Extracted skills from CV: {skills}")
                    
                    # If we still have no skills, try to extract from the raw resume text
                    if not skills:
                        from services.cv_refinement.jobs_suggestion import extract_skills_from_text
                        raw_skills = extract_skills_from_text(resume_text)
                        skills = set(skill.lower() for skill in raw_skills if skill)
                        logger.info(f"Extracted {len(skills)} skills from raw text")
                    
                    # Log final skills count
                    logger.info(f"Total skills extracted: {len(skills)}")
                    
                    # Ensure we have a result for the response
                    result = job_suggestions if isinstance(job_suggestions, dict) else {}
        except Exception as e:
            logger.warning(f"Error processing resume: {str(e)}")
            result["message"] = f"Warning: Could not process resume - {str(e)}"
        
        # Get all jobs with more fields for better matching
        try:
            jobs_collection = db["jobs"]
            
            # First, try to find jobs with matching titles from CV
            matched_title_jobs = []
            if job_titles:
                title_regex = "|".join([re.escape(title.lower()) for title in job_titles])
                matched_title_jobs = list(jobs_collection.find(
                    {"title": {"$regex": title_regex, "$options": "i"}},
                    {
                        "_id": 1,
                        "title": 1,
                        "companyName": 1,
                        "company": 1,
                        "required_skills": 1,
                        "preferred_skills": 1
                    }
                ).limit(20))
                logger.info(f"Found {len(matched_title_jobs)} jobs with matching titles")
            
            # Get other jobs if we don't have enough title matches
            remaining_slots = 20 - len(matched_title_jobs)
            other_jobs = []
            if remaining_slots > 0:
                other_jobs = list(jobs_collection.find(
                    {
                        "_id": {"$nin": [j["_id"] for j in matched_title_jobs] if matched_title_jobs else []}
                    },
                    {
                        "_id": 1,
                        "title": 1,
                        "companyName": 1,
                        "company": 1,
                        "required_skills": 1,
                        "preferred_skills": 1
                    }
                ).limit(remaining_slots))
            
            all_jobs = matched_title_jobs + other_jobs
            
            # Format jobs to match the expected output
            formatted_jobs = []
            for job in all_jobs:
                try:
                    # Get job details
                    title = job.get("title", "No Title").strip()
                    company = job.get("companyName", job.get("company", "Company Not Specified")).strip()
                    
                    # Calculate match scores (0-100%)
                    title_score = 0
                    required_skills = set(skill.lower() for skill in job.get("required_skills", []))
                    preferred_skills = set(skill.lower() for skill in job.get("preferred_skills", []))
                    
                    # 1. Title match (40% weight)
                    title_lower = title.lower()
                    title_score = 0
                    max_title_score = 40
                    
                    # Vietnamese to English title mapping for better matching
                    title_mapping = {
                        'lập trình viên': ['developer', 'programmer', 'engineer'],
                        'kỹ sư': ['engineer', 'developer'],
                        'chuyên viên': ['specialist', 'expert', 'officer'],
                        'nhân viên': ['staff', 'officer', 'associate']
                    }
                    
                    # Check each job title from CV against the job posting title
                    for jt in job_titles:
                        jt_lower = jt.lower()
                        
                        # 1. Exact match (100% score)
                        if jt_lower == title_lower:
                            title_score = max_title_score
                            break
                            
                        # 2. Check Vietnamese title mapping
                        matched = False
                        for vn_term, en_terms in title_mapping.items():
                            if vn_term in jt_lower and any(term in title_lower for term in en_terms):
                                title_score = max(title_score, max_title_score * 0.9)
                                matched = True
                                break
                        if matched:
                            continue
                            
                        # 3. Contains match (e.g., 'frontend' in 'senior frontend developer')
                        if jt_lower in title_lower or title_lower in jt_lower:
                            title_score = max(title_score, max_title_score * 0.8)
                            continue
                            
                        # 4. Word overlap with stemming
                        title_words = set(re.findall(r'\w+', title_lower))
                        jt_words = set(re.findall(r'\w+', jt_lower))
                        
                        # Remove common words that don't add meaning
                        common_words = {'senior', 'junior', 'lead', 'staff', 'i', 'ii', 'iii', 'iv', 'v'}
                        title_words = title_words - common_words
                        jt_words = jt_words - common_words
                        
                        if title_words and jt_words:
                            overlap = len(title_words.intersection(jt_words))
                            if overlap > 0:
                                score = max_title_score * (0.5 + (overlap / max(len(title_words), len(jt_words)) * 0.5))
                                title_score = max(title_score, score)
                    
                    # If still no match, check for technical role indicators
                    if title_score == 0:
                        tech_indicators = ['developer', 'engineer', 'programmer', 'lập trình', 'kỹ thuật', 'technical', 'code', 'software']
                        if any(indicator in title_lower for indicator in tech_indicators):
                            title_score = max_title_score * 0.3  # 30% for technical roles
                        else:
                            title_score = max_title_score * 0.1  # 10% for non-technical roles
                    
                    # 2. Required skills match (50% weight, reduced from 60%)
                    required_skills_matched = 0
                    required_skills_total = len(required_skills)
                    
                    # If no required skills defined, use default skills based on job title
                    if required_skills_total == 0:
                        default_skills = get_default_kills_for_title(title)
                        required_skills = default_skills
                        required_skills_total = len(required_skills)
                        logger.info(f"Using default skills for {title}: {required_skills}")
                    
                    if required_skills_total > 0:
                        # More flexible skill matching with scoring
                        matched_required = []
                        skill_scores = []
                        
                        for skill in required_skills:
                            if not skill:
                                continue
                                
                            skill_lower = skill.lower()
                            skill_score = 0
                            
                            # Check for direct match (full points)
                            if skill_lower in skills:
                                skill_score = 1.0
                            else:
                                # Check for partial matches
                                best_match_score = 0
                                for user_skill in skills:
                                    user_skill_lower = user_skill.lower()
                                    # Exact match
                                    if skill_lower == user_skill_lower:
                                        best_match_score = 1.0
                                        break
                                    # One is contained in the other
                                    elif skill_lower in user_skill_lower or user_skill_lower in skill_lower:
                                        # Longer the common substring, higher the score
                                        common = len(set(skill_lower).intersection(set(user_skill_lower)))
                                        total = max(len(skill_lower), len(user_skill_lower))
                                        score = common / total
                                        best_match_score = max(best_match_score, score * 0.8)  # Max 80% for partial matches
                                
                                skill_score = best_match_score
                            
                            if skill_score > 0.5:  # Only count if match is good enough
                                matched_required.append({
                                    'skill': skill,
                                    'match_score': skill_score,
                                    'matched_with': next((s for s in skills if skill_lower in s.lower() or s.lower() in skill_lower), None)
                                })
                            
                            skill_scores.append(skill_score)
                        
                        # Calculate weighted score (average of skill match scores)
                        required_score = 50 * (sum(skill_scores) / max(1, len(skill_scores))) if skill_scores else 0
                        required_skills_matched = len([s for s in skill_scores if s > 0.5])
                        
                        # Debug logging
                        logger.info(f"Job: {title}")
                        logger.info(f"Required skills: {required_skills}")
                        logger.info(f"Matched required skills: {matched_required}")
                        logger.info(f"Available skills: {skills}")
                    else:
                        required_score = 0
                        logger.info(f"Job {title} has no required skills defined")
                    
                    # 3. Preferred skills match (10% weight, reduced from 20%)
                    preferred_skills_matched = 0
                    preferred_skills_total = len(preferred_skills)
                    
                    # If no preferred skills defined, use an empty list
                    if preferred_skills_total == 0:
                        preferred_skills = []
                    
                    if preferred_skills_total > 0:
                        # Similar flexible matching as required skills
                        matched_preferred = []
                        for skill in preferred_skills:
                            if not skill:
                                continue
                            skill_lower = skill.lower()
                            if skill_lower in skills:
                                matched_preferred.append(skill)
                            else:
                                for user_skill in skills:
                                    if skill_lower in user_skill or user_skill in skill_lower:
                                        matched_preferred.append(skill)
                                        break
                        
                        preferred_skills_matched = len(matched_preferred)
                        preferred_score = (preferred_skills_matched / max(1, preferred_skills_total)) * 10
                        
                        # Debug logging
                        logger.info(f"Preferred skills: {preferred_skills}")
                        logger.info(f"Matched preferred skills: {matched_preferred}")
                    else:
                        preferred_score = 0
                        logger.info(f"Job {title} has no preferred skills defined")
                    
                    # Calculate total match percentage with weighted components
                    total_score = title_score + required_score + preferred_score
                    
                    # Apply bonus for jobs that match the user's primary role
                    primary_role = job_titles[0].lower() if job_titles else ''
                    if primary_role and any(word in title_lower for word in primary_role.split()):
                        total_score = min(100, total_score * 1.2)  # 20% bonus for primary role match
                    
                    match_percentage = min(100, max(0, round(total_score)))
                    
                    # Only include jobs with at least 30% match
                    if match_percentage < 30:
                        continue
                    
                    # Add visual indicator for match level
                    if match_percentage >= 80:
                        match_indicator = ""  # High match
                    elif match_percentage >= 50:
                        match_indicator = ""  # Medium match
                    else:
                        match_indicator = ""  # Low match
                    
                    # Format display title with match info and skill details
                    skill_info = []
                    if required_skills_total > 0:
                        skill_info.append(f"{required_skills_matched}/{required_skills_total} required")
                    if preferred_skills_total > 0:
                        skill_info.append(f"{preferred_skills_matched}/{preferred_skills_total} preferred")
                    
                    # Add visual indicator for match level
                    if match_percentage >= 80:
                        match_indicator = "🟢"
                    elif match_percentage >= 50:
                        match_indicator = "🟡"
                    else:
                        match_indicator = "🔴"
                        
                    skill_str = " (" + ", ".join(skill_info) + ")" if skill_info else ""
                    display_title = f"{match_indicator} {title} ({match_percentage}%{skill_str})"
                    
                    # Log final match details
                    logger.info(f"Final match for {title}: {match_percentage}% (Title: {title_score/40*100:.0f}%, "
                              f"Required: {required_score/40*100:.0f}%, Preferred: {preferred_score/20*100:.0f}%)")
                    
                    formatted_jobs.append({
                        "id": str(job.get("_id", "")),
                        "title": display_title,
                        "company": company,
                        "match_percentage": match_percentage,
                        "relevance": "high" if match_percentage >= 70 else "medium" if match_percentage >= 40 else "low",
                        "matched_skills": {
                            "required": required_skills_matched,
                            "preferred": preferred_skills_matched,
                            "total_required": len(required_skills),
                            "total_preferred": len(preferred_skills)
                        }
                    })
                except Exception as job_error:
                    logger.warning(f"Error formatting job {job.get('_id', 'unknown')}: {str(job_error)}")
            
            if formatted_jobs:
                result["matching_jobs"] = formatted_jobs
                result["total_matches"] = len(formatted_jobs)
                if not result.get("message"):
                    result["message"] = f"Found {len(formatted_jobs)} available jobs."
            else:
                result["message"] = "No jobs found in the database."
                
        except Exception as e:
            logger.error(f"Error fetching jobs from database: {str(e)}")
            if not result.get("message"):
                result["message"] = "Error fetching job listings. Please try again later."
        
        return result

    except Exception as e:
        error_msg = f"Error in jobs_suggestion: {str(e)}"
        logger.error(error_msg)
        
        # Try to return a helpful response even in case of error
        error_response = {
            "status": "error",
            "error": "An error occurred while processing your request",
            "details": str(e),
            "matching_jobs": [],
            "total_matches": 0
        }
        
        # Try to get at least some jobs to show
        try:
            db = get_database()
            jobs_collection = db["jobs"]
            all_jobs = list(jobs_collection.find(
                {},
                {
                    "_id": 1,
                    "title": 1,
                    "companyName": 1,
                    "company": 1,
                    "location": 1
                }
            ).limit(5))
            
            if all_jobs:
                formatted_jobs = [{
                    "id": str(job.get("_id", "")),
                    "title": job.get("title", "Job Title"),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "match_percentage": 0,
                    "relevance": "low",
                    "matched_skills": {
                        "required": 0,
                        "preferred": 0,
                        "total_required": 0,
                        "total_preferred": 0
                    }
                } for job in all_jobs]
                
                return {
                    "status": "success",
                    "matching_jobs": formatted_jobs,
                    "total_matches": len(formatted_jobs)
                }
                
            return {
                "status": "success",
                "matching_jobs": [],
                "total_matches": 0
            }
        except Exception as inner_e:
            logger.error(f"Error in fallback job fetch: {str(inner_e)}")
            
        # Return 200 with error details instead of 500 to show partial results
        return JSONResponse(
            status_code=200,
            content=error_response
        )


# ---------- Health / quick check ----------
@app.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# to run : uvicorn main:app --reload
# http://127.0.0.1:8000/docs