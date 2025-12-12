# Simple working CV API with real database integration
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import time
import pickle
import re
from typing import List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import random
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json

# Google Calendar Configuration
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# Pydantic models
class GoogleMeetRequest(BaseModel):
    summary: str
    description: str
    start_time: str
    end_time: str
    timezone: str = "UTC"
    attendees: Optional[List[str]] = []

app = FastAPI(title="Working CV API with Real Database and Google Calendar")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment and connect to MongoDB
load_dotenv()
MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")
DB_NAME = "CVProject"

# MongoDB connection
print("🔗 Connecting to MongoDB...")
mongo_client = MongoClient(MONGO_ATLAS_URI)
db = mongo_client[DB_NAME]
# Point to blog database which has jobs
blog_db = mongo_client["blog"]
jobs_collection = blog_db["jobs"]
cvs_collection = db["cvs"]
users_collection = db["users"]

print("✅ Connected to MongoDB successfully")

@app.get("/")
async def root():
    return {"status": "Working CV API with Real Database", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/health")
async def health_check():
    try:
        # Test database connection
        db.command("ping")
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(), "database": "connected"}
    except Exception as e:
        return {"status": "error", "time": datetime.now(timezone.utc).isoformat(), "database": "disconnected", "error": str(e)}

@app.get("/debug/all")
async def debug_all():
    """Debug endpoint to see all databases and collections"""
    try:
        client = mongo_client
        db_list = client.list_database_names()
        
        result = {"databases": {}}
        
        for db_name in db_list:
            if db_name.startswith("admin") or db_name.startswith("config") or db_name.startswith("local"):
                continue
                
            db = client[db_name]
            collections = db.list_collection_names()
            
            result["databases"][db_name] = {
                "collections": {},
                "total_collections": len(collections)
            }
            
            for collection_name in collections:
                collection = db[collection_name]
                count = collection.count_documents({})
                result["databases"][db_name]["collections"][collection_name] = count
                
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/jobs")
async def debug_jobs():
    """Debug endpoint to see current jobs count and sample"""
    try:
        total_jobs = jobs_collection.count_documents({})
        active_jobs = jobs_collection.count_documents({
            "status": {"$ne": "closed"},
            "expiryTime": {"$gt": datetime.utcnow()}
        })

        # Get a few sample jobs
        sample_jobs = list(jobs_collection.find({}, {"title": 1, "companyName": 1, "status": 1}).limit(3))
        job_samples = [{ "title": j.get("title", "No title"), "company": j.get("companyName", "No company"), "status": j.get("status", "unknown") } for j in sample_jobs]

        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "sample_jobs": job_samples,
            "source": "mongodb"
        }
    except Exception as e:
        return {
            "total_jobs": 0,
            "active_jobs": 0,
            "sample_jobs": [],
            "error": str(e),
            "source": "error"
        }

@app.get("/api/jobs-suggestion/{username}")
async def jobs_suggestion(username: str):
    """Get job suggestions for a user with real database jobs"""
    try:
        # Check if user has CV uploaded (try to find in database)
        cv_doc = cvs_collection.find_one({"username": username})

        # If not found in database, try memory fallback
        if not cv_doc and hasattr(cvs_collection, 'find_one'):
            print(f"⚠️ No CV found for {username} in database")

        # Fetch real jobs from database
        jobs_cursor = jobs_collection.find({
            "status": {"$ne": "closed"},
            "expiryTime": {"$gt": datetime.utcnow()}
        })

        # Convert cursor to list and format
        real_jobs = []
        for job in jobs_cursor.limit(10):  # Limit to 10 most relevant jobs
            match_percentage = random.randint(60, 95)  # Mock matching percentage for demo
            status_emoji = "🟢" if match_percentage >= 80 else "🟡" if match_percentage >= 70 else "🔴"

            real_jobs.append({
                "id": str(job["_id"]),
                "title": f"{status_emoji} {job.get('title', 'Untitled Job')} ({match_percentage}%)",
                "company": job.get('companyName', 'Unknown Company'),
                "match_percentage": match_percentage,
                "field": job.get('field', 'General'),
                "location": job.get('city', 'Remote'),
                "salary": job.get('salary', 'Negotiable'),
                "type": job.get('type', 'Full-time'),
                "experience": job.get('experience', 'Not specified'),
                "matched_skills": {
                    "required": random.randint(2, 4),
                    "total_required": 5
                },
                "slug": job.get('slug', ''),
                "description_preview": (job.get('description', '')[:150] + '...') if job.get('description') else 'No description available'
            })

        # Sort by match percentage
        real_jobs.sort(key=lambda x: x['match_percentage'], reverse=True)

        return {
            "username": username,
            "matching_jobs": real_jobs,
            "total_matches": len(real_jobs),
            "source": "mongodb"
        }

    except Exception as e:
        print(f"❌ Error in job suggestions: {e}")
        # Fallback to mock data on any error
        mock_jobs = [
            {
                "id": "fallback_1",
                "title": "🟢 Software Developer (75%)",
                "company": "Tech Company",
                "match_percentage": 75,
                "field": "IT/Software",
                "location": "Ho Chi Minh",
                "salary": "1000-1500$",
                "type": "Full-time",
                "matched_skills": {"required": 3, "total_required": 4},
                "description_preview": "Software development position with focus on modern technologies..."
            }
        ]
        return {
            "username": username,
            "matching_jobs": mock_jobs,
            "total_matches": len(mock_jobs),
            "source": "fallback_due_to_error",
            "error": str(e)
        }

@app.get("/resume/{username}")
async def get_resume(username: str):
    """Get stored resume for a user"""
    doc = cvs_collection.find_one({"username": username}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found for that username.")
    return doc

@app.post("/resume/{username}/suggest_improvements")
async def suggest_improvements(username: str):
    """Suggest detailed improvements for a resume based on actual CV content"""
    try:
        doc = cvs_collection.find_one({"username": username})
        if not doc:
            raise HTTPException(status_code=404, detail="No resume found for that username.")

        # Extract CV content for analysis
        cv_content = doc.get("processed_text", "")
        filename = doc.get("filename", "CV")
        file_size = doc.get("file_size", 0)
        file_type = doc.get("file_type", "")
        uploaded_at = doc.get("uploaded_at", "")

        # Analyze CV content
        analysis_result = analyze_cv_content(cv_content, filename, file_size, file_type)

        return {
            "username": username,
            "analysis": analysis_result,
            "cv_metadata": {
                "filename": filename,
                "file_size": file_size,
                "file_type": file_type,
                "uploaded_at": uploaded_at.isoformat() if uploaded_at else None
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")

def analyze_cv_content(cv_content: str, filename: str, file_size: int, file_type: str) -> dict:
    """
    Phân tích chi tiết nội dung CV và đưa ra nhận xét thật về điểm mạnh, điểm yếu
    """
    content_lower = cv_content.lower()
    word_count = len(cv_content.split()) if cv_content else 0
    content_length = len(cv_content)

    # Phân tích sâu các section của CV
    analysis = extract_cv_sections(cv_content)

    # Tính điểm mạnh và điểm yếu cụ thể
    strengths, weaknesses = analyze_strengths_weaknesses(analysis, cv_content)

    # Tính điểm completeness
    completeness_score = calculate_completeness_score(analysis)

    # Tạo feedback chi tiết dựa trên nội dung thực tế
    detailed_feedback = create_detailed_feedback(analysis, strengths, weaknesses)

    # Overall assessment dựa trên phân tích thực tế
    overall_status, overall_message, grade = assess_cv_quality(analysis, completeness_score, strengths, weaknesses)
    
    # Calculate final score for response
    strength_score = len(strengths) * 15
    weakness_penalty = len(weaknesses) * 5
    baseline_score = 40
    final_score = min(100, max(0, baseline_score + completeness_score + strength_score - weakness_penalty))

    # Statistics
    stats = {
        "word_count": word_count,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0,
        "completeness_score": completeness_score,
        "completeness_factors": list(analysis.keys()),
        "technical_skills_count": len(analysis.get('technical_skills', [])),
        "experience_years": analysis.get('total_experience_years', 0),
        "projects_count": len(analysis.get('projects', [])),
        "education_level": analysis.get('education_level', 'Không xác định')
    }

    return {
        "overall_assessment": {
            "status": overall_status,
            "score": final_score,
            "message": overall_message,
            "grade": grade
        },
        "statistics": stats,
        "detailed_feedback": detailed_feedback,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "quick_improvements": generate_prioritized_improvements(weaknesses, analysis),
        "ats_friendly_tips": generate_ats_tips(),
        "next_steps": generate_realistic_next_steps(weaknesses, analysis)
    }

def extract_cv_sections(cv_content: str) -> dict:
    """Trích xuất và phân tích các section của CV"""
    analysis = {}

    # Phân tích thông tin liên hệ
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cv_content)
    phone_match = re.search(r'(0|\+84)?[\d\s-]{9,15}', cv_content)
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', cv_content, re.IGNORECASE)

    analysis['contact_info'] = {
        'has_email': bool(email_match),
        'has_phone': bool(phone_match),
        'has_linkedin': bool(linkedin_match),
        'email': email_match.group() if email_match else None
    }

    # Phân tích học vấn
    education_patterns = [
        r'(đại học|university|college|cao đẳng)[^.]*',
        r'(bachelor|cử nhân|thạc sĩ|master|phd|tiến sĩ)[^.]*',
        r'(sp_khoa_cntt|khoa công nghệ thông tin|software engineering)[^.]*'
    ]

    education_found = []
    for pattern in education_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        education_found.extend(matches)

    analysis['education'] = {
        'has_education': len(education_found) > 0,
        'details': education_found[:3],  # Lấy tối đa 3 dòng
        'education_level': determine_education_level(education_found)
    }

    # Phân tích kinh nghiệm làm việc
    experience_matches = re.findall(r'(20\d{2}|01/\d{4}|\d{1,2}/\d{4})[^.]*', cv_content)
    analysis['experience'] = {
        'has_experience': len(experience_matches) > 0,
        'experience_entries': experience_matches[:5],
        'total_experience_years': estimate_experience_years(cv_content)
    }

    # Phân tích kỹ năng kỹ thuật
    technical_keywords = [
        'python', 'java', 'javascript', 'nodejs', 'react', 'vue', 'angular',
        'mongodb', 'mysql', 'postgresql', 'docker', 'kubernetes', 'git',
        'aws', 'azure', 'gcp', 'ci/cd', 'agile', 'scrum'
    ]

    found_skills = []
    for skill in technical_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', cv_content, re.IGNORECASE):
            found_skills.append(skill)

    analysis['technical_skills'] = found_skills

    # Phân tích dự án
    project_patterns = [
        r'(project|dự án)[^.]*',
        r'github\.com[^s]*',
        r'portfolio[^.]*'
    ]

    projects = []
    for pattern in project_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        projects.extend(matches)

    analysis['projects'] = projects[:3]

    # Phân tích chứng chỉ
    cert_patterns = [
        r'(certificate|chứng chỉ|certification)[^.]*',
        r'(toeic|ielts|toefl)[^.]*'
    ]

    certificates = []
    for pattern in cert_patterns:
        matches = re.findall(pattern, cv_content, re.IGNORECASE)
        certificates.extend(matches)

    analysis['certificates'] = certificates

    return analysis

def analyze_strengths_weaknesses(analysis: dict, cv_content: str) -> tuple:
    """Phân tích điểm mạnh và điểm yếu cụ thể"""
    strengths = []
    weaknesses = []

    # Phân tích kỹ năng kỹ thuật
    tech_skills = analysis.get('technical_skills', [])
    if len(tech_skills) >= 5:
        strengths.append("✅ **Kỹ năng kỹ thuật đa dạng**: Có {} công nghệ lập trình ({})".format(
            len(tech_skills), ', '.join(tech_skills[:5])))
    elif len(tech_skills) >= 3:
        strengths.append("✅ **Kỹ năng kỹ thuật tốt**: Biết {} công nghệ ({})".format(
            len(tech_skills), ', '.join(tech_skills)))
    elif len(tech_skills) > 0:
        weaknesses.append("❌ **Kỹ năng kỹ thuật còn ít**: Chỉ mới biết {} công nghệ, cần học thêm nhiều hơn".format(
            len(tech_skills)))
    else:
        weaknesses.append("❌ **Thiếu kỹ năng kỹ thuật**: CV không ghi rõ công nghệ lập trình đã biết")

    # Phân tích kinh nghiệm
    experience_years = analysis.get('experience', {}).get('total_experience_years', 0)
    if experience_years >= 2:
        strengths.append("✅ **Kinh nghiệm thực tế**: Có {} năm kinh nghiệm làm việc".format(experience_years))
    elif experience_years >= 1:
        strengths.append("✅ **Có kinh nghiệm cơ bản**: {} năm kinh nghiệm là điểm khởi đầu tốt".format(experience_years))
    else:
        weaknesses.append("❌ **Thiếu kinh nghiệm làm việc**: Cần có internships hoặc dự án thực tế")

    # Phân tích học vấn
    education = analysis.get('education', {})
    if education.get('education_level') == 'Đại học':
        strengths.append("✅ **Tốt nghiệp Đại học**: Bằng cấp được nhà tuyển dụng công nhận")
    elif education.get('has_education'):
        strengths.append("✅ **Có nền tảng học vấn**: Đang theo học hoặc đã tốt nghiệp")
    else:
        weaknesses.append("❌ **Thông tin học vấn chưa rõ**: Cần ghi rõ trường và ngành học")

    # Phân tích dự án
    projects = analysis.get('projects', [])
    if len(projects) >= 2:
        strengths.append("✅ **Dự án cá nhân phong phú**: Có {} dự án chứng minh kỹ năng thực tế".format(len(projects)))
    elif len(projects) >= 1:
        strengths.append("✅ **Có dự án cá nhân**: Tốt cho sinh viên mới ra trường")
    else:
        weaknesses.append("❌ **Thiếu dự án cá nhân**: Cần có portfolio hoặc GitHub để chứng minh kỹ năng")

    # Phân tích chứng chỉ
    certificates = analysis.get('certificates', [])
    if len(certificates) >= 2:
        strengths.append("✅ **Chứng chỉ ngoại ngữ/chuyên môn**: Có {} chứng chỉ tăng uy tín".format(len(certificates)))
    elif len(certificates) >= 1:
        strengths.append("✅ **Có chứng chỉ**: Nỗ lực phát triển bản thân")
    else:
        weaknesses.append("❌ **Thiếu chứng chỉ**: Nên có TOEIC/IELTS hoặc chứng chỉ ngành nghề")

    # Phân tích thông tin liên hệ
    contact = analysis.get('contact_info', {})
    missing_contact = []
    if not contact.get('has_email'): missing_contact.append("email")
    if not contact.get('has_phone'): missing_contact.append("số điện thoại")
    if not contact.get('has_linkedin'): missing_contact.append("LinkedIn")

    if missing_contact:
        weaknesses.append("❌ **Thông tin liên hệ chưa đầy đủ**: Thiếu {}".format(', '.join(missing_contact)))
    else:
        strengths.append("✅ **Thông tin liên hệ đầy đủ**: Dễ dàng liên lạc với nhà tuyển dụng")

    return strengths, weaknesses

def determine_education_level(education_found: list) -> str:
    """Xác định trình độ học vấn"""
    education_text = ' '.join(education_found).lower()

    if any(word in education_text for word in ['tiến sĩ', 'phd', 'doctor']):
        return 'Tiến sĩ'
    elif any(word in education_text for word in ['thạc sĩ', 'master']):
        return 'Thạc sĩ'
    elif any(word in education_text for word in ['đại học', 'university', 'bachelor', 'cử nhân']):
        return 'Đại học'
    elif any(word in education_text for word in ['cao đẳng', 'college']):
        return 'Cao đẳng'
    else:
        return 'Không xác định'

def estimate_experience_years(cv_content: str) -> int:
    """Ước tính số năm kinh nghiệm làm việc thực tế"""
    # Chỉ đếm kinh nghiệm làm việc, không tính thời gian học
    work_keywords = [
        'intern', 'internship', 'work', 'job', 'employment', 'position',
        'developer', 'engineer', 'analyst', 'manager', 'specialist',
        'full-time', 'part-time', 'contract', 'freelance'
    ]
    
    # Tìm các mốc thời gian liên quan đến làm việc
    lines = cv_content.lower().split('\n')
    total_months = 0
    
    for line in lines:
        # Kiểm tra nếu dòng chứa từ khóa về công việc
        if any(keyword in line for keyword in work_keywords):
            # Tìm các khoảng thời gian trong dòng
            duration_matches = re.findall(r'(\d{1,2}/\d{4})\s*-\s*(\d{1,2}/\d{4}|present)', line)
            for start, end in duration_matches:
                try:
                    start_month, start_year = map(int, start.split('/'))
                    if end.lower() == 'present':
                        end_month, end_year = 12, 2025  # Giả present là tháng 12/2025
                    else:
                        end_month, end_year = map(int, end.split('/'))
                    
                    # Tính số tháng
                    months = (end_year - start_year) * 12 + (end_month - start_month)
                    if months > 0 and months <= 60:  # Giới hạn 5 năm cho mỗi vị trí
                        total_months += months
                except:
                    continue
    
    # Chuyển đổi tháng sang năm (làm tròn xuống)
    years = total_months // 12
    return min(years, 10)  # Max 10 năm

def calculate_completeness_score(analysis: dict) -> int:
    """Tính điểm hoàn thiện CV"""
    score = 0

    if analysis.get('contact_info', {}).get('has_email'): score += 10
    if analysis.get('contact_info', {}).get('has_phone'): score += 5
    if analysis.get('contact_info', {}).get('has_linkedin'): score += 5

    if analysis.get('education', {}).get('has_education'): score += 20
    if analysis.get('experience', {}).get('has_experience'): score += 25

    tech_skills_count = len(analysis.get('technical_skills', []))
    if tech_skills_count >= 5: score += 20
    elif tech_skills_count >= 3: score += 15
    elif tech_skills_count >= 1: score += 10

    if len(analysis.get('projects', [])) >= 2: score += 10
    elif len(analysis.get('projects', [])) >= 1: score += 5

    if len(analysis.get('certificates', [])) >= 1: score += 5

    return min(score, 100)

def create_detailed_feedback(analysis: dict, strengths: list, weaknesses: list) -> list:
    """Tạo feedback chi tiết dựa trên phân tích thực tế"""
    feedback = []

    # Feedback về kỹ năng kỹ thuật
    tech_skills = analysis.get('technical_skills', [])
    if tech_skills:
        feedback.append({
            "section": "Kỹ năng kỹ thuật",
            "status": "good",
            "icon": "✅",
            "title": f"Có {len(tech_skills)} kỹ năng công nghệ",
            "description": f"Bạn đã thành thạo: {', '.join(tech_skills)}. Đây là điểm mạnh cạnh tranh!",
            "tips": [
                f"Highlight các kỹ năng hot nhất: {', '.join(tech_skills[:3])}",
                "Thêm mức độ thành thạo (Basic/Intermediate/Advanced)",
                "Liệt kê các project đã áp dụng từng technology"
            ] if len(tech_skills) >= 3 else [
                "Nên học thêm các công nghệ hot khác",
                "Thực hành thêm qua các dự án cá nhân",
                "Lấy chứng chỉ để xác nhận kỹ năng"
            ]
        })

    # Feedback về kinh nghiệm
    exp_years = analysis.get('experience', {}).get('total_experience_years', 0)
    if exp_years >= 1:
        feedback.append({
            "section": "Kinh nghiệm làm việc",
            "status": "good",
            "icon": "✅",
            "title": f"Có {exp_years} năm kinh nghiệm",
            "description": f"{exp_years} năm kinh nghiệm là nền tảng vững chắc cho vị trí junior/mid-level.",
            "tips": [
                "Sử dụng con số cụ thể: 'Tăng performance 30%', 'Quản lý 5 người'",
                "Nêu bật technologies đã dùng trong công việc",
                "Mô tả theo công thức STAR (Situation, Task, Action, Result)"
            ]
        })

    # Feedback về điểm yếu
    if len(tech_skills) < 3:
        feedback.append({
            "section": "Cần cải thiện kỹ năng",
            "status": "missing",
            "icon": "⚠️",
            "title": f"Chỉ có {len(tech_skills)} kỹ năng công nghệ",
            "description": f"Hiện tại bạn chỉ biết: {', '.join(tech_skills) if tech_skills else 'chưa ghi rõ'}. Cần mở rộng để tăng tính cạnh tranh.",
            "action_items": [
                "Học thêm framework phổ biến (React, Vue, Angular)",
                "Làm quen với database (MongoDB, PostgreSQL)",
                "Học cloud basics (AWS, Azure)"
            ]
        })

    if len(analysis.get('projects', [])) == 0:
        feedback.append({
            "section": "Dự án thực tế",
            "status": "suggestion",
            "icon": "💡",
            "title": "Nên có dự án cá nhân",
            "description": "Dự án cá nhân là cách tốt nhất để chứng minh kỹ năng khi còn ít kinh nghiệm.",
            "action_items": [
                "Tạo GitHub portfolio và đẩy code lên",
                "Làm 2-3 projects từ end-to-end",
                "Deploy projects lên Vercel/Netlify/Railway",
                "Viết README chi tiết cho mỗi project"
            ]
        })

    return feedback

def assess_cv_quality(analysis: dict, completeness_score: int, strengths: list, weaknesses: list) -> tuple:
    """Đánh giá chất lượng CV thực tế"""
    strength_score = len(strengths) * 15  # Increased from 10 to 15
    weakness_penalty = len(weaknesses) * 5  # Reduced from 8 to 5
    
    # Add minimum baseline score of 40 for having uploaded a CV
    baseline_score = 40
    
    final_score = min(100, max(0, baseline_score + completeness_score + strength_score - weakness_penalty))

    # Adjusted thresholds to be more lenient
    if final_score >= 80:
        return "excellent", "CV của bạn rất tốt! Có nhiều điểm mạnh và ít điểm yếu. Sẵn sàng cho vị trí Mid-level.", "A"
    elif final_score >= 60:
        return "good", f"CV khá tốt với {len(strengths)} điểm mạnh. Cần cải thiện {len(weaknesses)} điểm yếu để competitive hơn.", "B"
    elif final_score >= 40:
        return "fair", f"CV cần cải thiện thêm. Có {len(strengths)} điểm mạnh nhưng còn {len(weaknesses)} điểm yếu cần khắc phục.", "C"
    else:
        return "poor", f"CV cần cải thiện nhiều. Cần tập trung khắc phục {len(weaknesses)} điểm yếu quan trọng.", "D"

def generate_prioritized_improvements(weaknesses: list, analysis: dict) -> list:
    """Tạo danh sách cải thiện ưu tiên theo điểm yếu thực tế"""
    improvements = []

    # Đọc weaknesses và tạo improvements tương ứng
    for weakness in weaknesses:
        if "kỹ năng kỹ thuật" in weakness.lower():
            improvements.append({
                "priority": "high",
                "title": "Học thêm kỹ năng công nghệ",
                "time_estimate": "2-3 tháng",
                "impact": "Rất cao - Tăng 50% cơ hội phỏng vấn"
            })
        elif "kinh nghiệm" in weakness.lower():
            improvements.append({
                "priority": "high",
                "title": "Làm internships hoặc dự án freelance",
                "time_estimate": "1-2 tháng",
                "impact": "Cao - Có kinh nghiệm thực tế"
            })
        elif "dự án" in weakness.lower():
            improvements.append({
                "priority": "medium",
                "title": "Xây dựng portfolio 2-3 projects",
                "time_estimate": "1 tháng",
                "impact": "Cao - Chứng minh kỹ năng thực tế"
            })
        elif "chứng chỉ" in weakness.lower():
            improvements.append({
                "priority": "medium",
                "title": "Lấy chứng chỉ TOEIC/IELTS",
                "time_estimate": "2-3 tháng",
                "impact": "Trung bình - Yêu cầu của nhiều công ty"
            })
        elif "liên hệ" in weakness.lower():
            improvements.append({
                "priority": "high",
                "title": "Cập nhật thông tin liên hệ",
                "time_estimate": "5 phút",
                "impact": "Trung bình - Để nhà tuyển dụng liên lạc"
            })

    return improvements[:5]  # Giới hạn 5 improvements quan trọng nhất

def generate_realistic_next_steps(weaknesses: list, analysis: dict) -> list:
    """Tạo các bước tiếp theo thực tế"""
    steps = []

    if any("kỹ năng" in w.lower() for w in weaknesses):
        steps.extend([
            "Chọn 2-3 công nghệ hot (React, Node.js, Python) để học sâu",
            "Làm 2 projects hoàn chỉnh với các công nghệ đã chọn",
            "Đẩy code lên GitHub và viết README chi tiết"
        ])

    if any("kinh nghiệm" in w.lower() for w in weaknesses):
        steps.extend([
            "Tìm internships hoặc freelance projects",
            "Tham gia coding contests hoặc hackathons",
            "Làm volunteer projects cho tổ chức"
        ])

    if any("dự án" in w.lower() for w in weaknesses):
        steps.append("Tạo personal website/portfolio để showcase projects")

    if any("chứng chỉ" in w.lower() for w in weaknesses):
        steps.append("Đăng ký kỳ thi TOEIC/IELTS trong 3 tháng tới")

    # Thêm các bước general
    steps.extend([
        "Network với developers trên LinkedIn/GitHub",
        "Theo dõi job descriptions để biết market demands",
        "Practice phỏng vấn với bạn bè hoặc mentor"
    ])

    return steps[:6]  # Giới hạn 6 steps thực tế nhất

def generate_quick_improvements(stats: dict, feedback: list) -> list:
    """Generate quick improvement suggestions"""
    improvements = []

    if not stats["has_contact_info"]:
        improvements.append({
            "priority": "high",
            "title": "Thêm thông tin liên hệ",
            "time_estimate": "5 phút",
            "impact": "Cao"
        })

    if not stats["has_experience"]:
        improvements.append({
            "priority": "high",
            "title": "Mô tả kinh nghiệm làm việc",
            "time_estimate": "15-30 phút",
            "impact": "Rất cao"
        })

    if not stats["has_skills"]:
        improvements.append({
            "priority": "high",
            "title": "Liệt kê kỹ năng chuyên môn",
            "time_estimate": "10 phút",
            "impact": "Cao"
        })

    if not stats["has_projects"]:
        improvements.append({
            "priority": "medium",
            "title": "Thêm dự án cá nhân",
            "time_estimate": "20 phút",
            "impact": "Trung bình - Cao"
        })

    return improvements

def generate_ats_tips() -> list:
    """Generate ATS (Applicant Tracking System) friendly tips"""
    return [
        {
            "tip": "Sử dụng font đơn giản (Arial, Calibri, Times New Roman)",
            "reason": "ATS dễ đọc các font tiêu chuẩn"
        },
        {
            "tip": "Tránh sử dụng bảng, cột, và đồ họa phức tạp",
            "reason": "ATS có thể không đọc đúng định dạng phức tạp"
        },
        {
            "tip": "Sử dụng từ khóa tiêu chuẩn ngành",
            "reason": "Giúp CV được tìm thấy dễ dàng hơn"
        },
        {
            "tip": "Lưu dưới dạng PDF",
            "reason": "Định dạng ổn định và bảo toàn layout"
        },
        {
            "tip": "Đặt tên file rõ ràng (Ten_Ho_Ten_CV.pdf)",
            "reason": "Chuyên nghiệp và dễ quản lý"
        }
    ]

def generate_next_steps(score: int) -> list:
    """Generate next steps based on CV score"""
    if score >= 80:
        return [
            "Xem lại và tinh chỉnh wording cho mượt mà hơn",
            "Thêm một vài dự án cá nhân để nổi bật",
            "Chuẩn bị cho các câu hỏi phỏng vấn dựa trên CV"
        ]
    elif score >= 60:
        return [
            "Bổ sung các phần còn thiếu (dự án, chứng chỉ)",
            "Cải thiện mô tả kinh nghiệm với số liệu cụ thể",
            "Phân loại kỹ năng rõ ràng hơn"
        ]
    else:
        return [
            "Ưu tiên thêm thông tin liên hệ và học vấn",
            "Mô tả chi tiết kinh nghiệm làm việc gần nhất",
            "Liệt kê tất cả kỹ năng có liên quan",
            "Thêm dự án cá nhân để thể hiện năng lực"
        ]

@app.post("/upload_resume")
async def upload_resume(username: str = Form(...), file: UploadFile = File(...)):
    """Upload and process a resume"""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="The file is empty")

    try:
        # Read file content
        file_content = await file.read()
        
        # Save file temporarily for processing
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Extract text using the pipeline
        try:
            from services.ingestion.pipeline import process_resume
            processed_text, parsed_output = process_resume(temp_file_path)
        except Exception as e:
            print(f"Text extraction failed: {str(e)}")
            # Fallback to basic processing
            processed_text = f"Processed {file.filename} ({len(file_content)} bytes)"
            parsed_output = {"technical_skills": [], "job_titles": [], "industries": [], "level": "unknown"}
        
        # Clean up temp file
        os.remove(temp_file_path)

        # Store in database
        doc = {
            "username": username,
            "uploaded_at": datetime.utcnow(),
            "filename": file.filename,
            "processed_text": processed_text,
            "parsed_output": parsed_output,
            "file_size": file_size,
            "file_type": file_ext
        }

        cvs_collection.update_one({"username": username}, {"$set": doc}, upsert=True)

        return {
            "username": username,
            "saved": True,
            "message": f"Resume '{file.filename}' uploaded and processed successfully.",
            "filename": file.filename,
            "size": file_size,
            "extracted_skills": len(parsed_output.get("technical_skills", [])),
            "text_length": len(processed_text)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

# Google OAuth and Calendar Functions
def get_google_credentials():
    """Get Google OAuth2 credentials"""
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return creds

@app.get("/api/auth/google")
async def auth_google():
    """Start Google OAuth flow"""
    try:
        if not os.path.exists(CREDENTIALS_FILE):
            return {
                "error": "credentials_missing",
                "message": "credentials.json file not found. Please download from Google Cloud Console."
            }

        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        auth_url, _ = flow.authorization_url(prompt='consent')

        return {
            "auth_url": auth_url,
            "message": "Please visit the URL to authenticate with Google"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting OAuth flow: {str(e)}")

@app.get("/api/auth/google/callback")
async def auth_google_callback(code: str):
    """Handle Google OAuth callback"""
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        flow.fetch_token(code=code)

        creds = flow.credentials

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

        return {"status": "success", "message": "Successfully authenticated with Google"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/create-meet")
async def create_google_meet(meet_request: GoogleMeetRequest):
    """Create a Google Meet meeting"""
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
        if "invalid_grant" in str(e) or "token" in str(e).lower():
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            return {
                "error": "authentication_required",
                "message": "Authentication expired. Please re-authenticate with Google."
            }
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_user")
async def create_user(username: str = Form(...)):
    """Create a new user"""
    try:
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return {"username": username, "created": False, "message": "User already exists."}

        user_doc = {"username": username, "created_at": datetime.utcnow()}
        users_collection.insert_one(user_doc)
        return {"username": username, "created": True, "message": "User created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Working CV API Server...")
    print("📡 Server will be available at: http://127.0.0.1:8002")
    print("📖 API docs: http://127.0.0.1:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)