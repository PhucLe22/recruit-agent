# Simple FastAPI server for testing CV upload functionality
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

app = FastAPI(title="CV Upload Test API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment
load_dotenv()
MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")
DB_NAME = "CVProject"
USERS_COLLECTION = "users"
CVS_COLLECTION = "cvs"

# MongoDB connection
try:
    mongo_client = MongoClient(MONGO_ATLAS_URI)
    db = mongo_client[DB_NAME]
    users_collection = db[USERS_COLLECTION]
    cvs_collection = db[CVS_COLLECTION]
    print("✅ Connected to MongoDB successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    mongo_client = None
    db = None

@app.get("/")
async def root():
    return {"status": "CV Upload Test API is running", "time": datetime.utcnow().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        if db:
            db.command("ping")
        return {"status": "ok", "time": datetime.utcnow().isoformat(), "database": "connected"}
    except Exception as e:
        return {"status": "error", "time": datetime.utcnow().isoformat(), "database": "disconnected", "error": str(e)}

@app.post("/create_user")
async def create_user(username: str = Form(...)):
    """Create a new user"""
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        now = datetime.utcnow()
        user_doc = {"username": username, "created_at": now}

        # Check if user exists
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return {"username": username, "created": False, "message": "User already exists."}

        # Create new user
        users_collection.insert_one(user_doc)
        return {"username": username, "created": True, "message": "User created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.post("/upload_resume")
async def upload_resume(username: str = Form(...), file: UploadFile = File(...)):
    """Upload and process a resume"""
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

    # Check file size (max 5MB)
    max_size = 5 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is 5MB. Your file is {file_size/1024/1024:.2f}MB")

    if file_size == 0:
        raise HTTPException(status_code=400, detail="The file is empty")

    try:
        # Create user if not exists
        if users_collection.find_one({"username": username}) is None:
            users_collection.insert_one({
                "username": username,
                "email": f"{username}@cvproject.com",
                "created_at": datetime.utcnow()
            })

        # Read file content (for demo, just store filename and basic info)
        file_content = await file.read()
        processed_text = f"Demo processing for {file.filename} ({len(file_content)} bytes)"

        # Mock parsed output
        parsed_output = {
            "filename": file.filename,
            "size": file_size,
            "type": file_ext,
            "processed_at": datetime.utcnow().isoformat()
        }

        # Store in database
        doc = {
            "username": username,
            "uploaded_at": datetime.utcnow(),
            "filename": file.filename,
            "processed_text": processed_text,
            "parsed_output": parsed_output,
        }

        result = cvs_collection.update_one({"username": username}, {"$set": doc}, upsert=True)

        return {
            "username": username,
            "saved": True,
            "message": f"Resume '{file.filename}' uploaded and stored successfully.",
            "filename": file.filename,
            "size": file_size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

@app.get("/resume/{username}")
async def get_resume(username: str):
    """Get stored resume for a user"""
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Database not connected")

    doc = cvs_collection.find_one({"username": username}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found for that username.")
    return doc

@app.post("/resume/{username}/suggest_improvements")
async def suggest_improvements(username: str):
    """Suggest improvements for a resume"""
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Database not connected")

    doc = cvs_collection.find_one({"username": username})
    if not doc or "processed_text" not in doc:
        raise HTTPException(status_code=404, detail="No resume found for that username.")

    # Mock improvements
    improvements = [
        {
            "category": "Structure",
            "suggestion": "Add a professional summary at the beginning of your CV"
        },
        {
            "category": "Skills",
            "suggestion": "Include more technical skills with proficiency levels"
        },
        {
            "category": "Experience",
            "suggestion": "Quantify your achievements with specific metrics"
        }
    ]

    return {
        "username": username,
        "improvements": improvements
    }

@app.get("/api/jobs-suggestion/{username}")
async def jobs_suggestion(username: str):
    """Get job suggestions for a user"""
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Database not connected")

    doc = cvs_collection.find_one({"username": username})
    if not doc:
        raise HTTPException(status_code=404, detail="User CV not found.")

    # Mock job suggestions
    mock_jobs = [
        {
            "id": "1",
            "title": "🟢 Frontend Developer (85%)",
            "company": "Tech Company A",
            "match_percentage": 85,
            "matched_skills": {
                "required": 3,
                "total_required": 4
            }
        },
        {
            "id": "2",
            "title": "🟡 Full Stack Developer (65%)",
            "company": "Startup B",
            "match_percentage": 65,
            "matched_skills": {
                "required": 2,
                "total_required": 3
            }
        }
    ]

    return {
        "matching_jobs": mock_jobs,
        "total_matches": len(mock_jobs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)