import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.llm_utils import get_ollama_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Skill categories for matching
SKILL_CATEGORIES = {
    'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
    'frontend': ['react', 'vue', 'angular', 'html', 'css', 'tailwind', 'next.js', 'redux'],
    'backend': ['node.js', 'django', 'flask', 'express', 'spring boot', 'fastapi'],
    'databases': ['mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
    'ai_ml': ['tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy']
}

# Flatten all skills for easy lookup
ALL_SKILLS = {skill for skills in SKILL_CATEGORIES.values() for skill in skills}

# Matching weights
WEIGHTS = {
    'title': 0.3,
    'skills': 0.4,
    'experience': 0.2,
    'industry': 0.1
}


def extract_skills_and_experience(resume_text: str, model_name: str = "qwen2.5:3b") -> dict:
    """Extract skills and experience from resume using LLM."""
    prompt = f"""Extract from this resume in JSON format:
    {{
        "technical_skills": [list of skills],
        "job_titles": [list of job titles],
        "industries": [list of industries],
        "experience_level": "entry|mid|senior|executive",
        "education_level": "highest degree"
    }}
    
    Resume: {resume_text}
    """
    
    try:
        response = get_ollama_response(prompt=prompt, model=model_name)
        # Clean JSON from markdown
        result_text = re.sub(r'```(?:json)?\n?|\n?```', '', response.strip())
        result = json.loads(result_text)
        
        # Normalize skills
        if isinstance(result.get('technical_skills'), list):
            result['technical_skills'] = [
                {'name': s if isinstance(s, str) else s.get('name', ''), 
                 'experience_years': 0 if isinstance(s, str) else s.get('experience_years', 0)}
                for s in result['technical_skills']
            ]
        
        return result
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        # Fallback to keyword extraction
        try:
            from services.cv_refinement.keyword_extraction import extract_keywords_from_resume
            fallback_result = extract_keywords_from_resume(resume_text)
            return {
                "technical_skills": fallback_result.get("technical_skills", []),
                "job_titles": fallback_result.get("job_titles", []),
                "industries": fallback_result.get("industries", []),
                "experience_level": fallback_result.get("level", "mid"),
                "education_level": ""
            }
        except Exception as fallback_error:
            logger.error(f"Fallback extraction also failed: {fallback_error}")
            return {
                "technical_skills": [],
                "job_titles": [],
                "industries": [],
                "experience_level": "mid",
                "education_level": ""
            }


def extract_skills_from_text(text: str) -> Set[str]:
    """Fast skill extraction using regex patterns."""
    text_lower = text.lower()
    found_skills = set()
    
    for skill in ALL_SKILLS:
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            found_skills.add(skill)
    
    # Extract special patterns (C++, C#, .NET, etc.)
    special_patterns = [r'\b\w+\+\+\b', r'\b\w+#\b', r'\.\w+\b']
    for pattern in special_patterns:
        found_skills.update(m.group(0).lower() for m in re.finditer(pattern, text_lower))
    
    return found_skills


def calculate_match_score(job: Dict, resume_skills: Set[str], resume_titles: List[str], 
                         resume_level: str) -> float:
    """Calculate job match score efficiently."""
    score = 0.0
    
    # 1. Title matching (30%)
    job_title = job.get('title', '').lower()
    if any(title.lower() in job_title for title in resume_titles):
        score += WEIGHTS['title']
    
    # 2. Skills matching (40%)
    job_skills = set()
    for field in ['skills', 'required_skills', 'technique']:
        if field in job:
            if isinstance(job[field], list):
                job_skills.update(s.lower() for s in job[field])
            elif isinstance(job[field], str):
                job_skills.update(s.strip().lower() for s in job[field].split(','))
    
    if job_skills:
        matched_skills = resume_skills & job_skills
        skill_ratio = len(matched_skills) / max(len(job_skills), 1)
        score += skill_ratio * WEIGHTS['skills']
    
    # 3. Experience level (20%)
    exp_levels = {'entry': 1, 'mid': 2, 'senior': 3, 'executive': 4}
    job_level = exp_levels.get(job.get('experience_level', '').lower(), 2)
    resume_level_num = exp_levels.get(resume_level.lower(), 2)
    
    if resume_level_num >= job_level:
        score += WEIGHTS['experience']
    else:
        score += WEIGHTS['experience'] * (resume_level_num / job_level)
    
    # 4. Industry bonus (10%)
    if job.get('industries'):
        score += WEIGHTS['industry']
    
    return round(score, 4)


def get_matching_jobs(
    resume_data: dict,
    limit: int = 10,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Find matching jobs with pagination and filtering.
    
    Returns: {jobs, total_matches, page, total_pages, has_next, has_previous}
    """
    try:
        # Extract resume data
        skills = {
            skill['name'].lower(): skill.get('experience_years', 0)
            for skill in resume_data.get('technical_skills', [])
            if isinstance(skill, dict) and 'name' in skill
        }
        
        job_titles = [t.lower() for t in resume_data.get('job_titles', []) if t]
        level = resume_data.get('experience_level', 'mid').lower()
        
        if not skills and not job_titles:
            return {
                'jobs': [], 'total_matches': 0, 'page': page, 
                'total_pages': 0, 'has_next': False, 'has_previous': False
            }
        
        # Connect to MongoDB
        mongo_uri = os.getenv("MONGO_ATLAS_URI")
        if not mongo_uri:
            raise ValueError("MONGO_ATLAS_URI not found")
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["CVProject"]
        jobs_collection = db["jobs"]
        
        # Build query
        query = {}
        
        # Skills or title search
        if skills or job_titles:
            search_terms = list(skills.keys())[:10] + job_titles[:5]
            query['$or'] = [
                {'title': {'$regex': '|'.join(re.escape(t) for t in job_titles), '$options': 'i'}},
                {'skills': {'$in': list(skills.keys())}},
                {'required_skills': {'$in': list(skills.keys())}}
            ]
        
        # Add filters
        if location:
            query['location'] = {'$regex': re.escape(location), '$options': 'i'}
        if experience_level:
            query['experience_level'] = {'$regex': f'^{re.escape(experience_level)}$', '$options': 'i'}
        
        # Get total count and execute query
        total_matches = jobs_collection.count_documents(query)
        total_pages = (total_matches + page_size - 1) // page_size
        skip = (page - 1) * page_size
        
        jobs = list(jobs_collection.find(query).skip(skip).limit(page_size))
        
        # Calculate match scores
        for job in jobs:
            job['match_score'] = calculate_match_score(
                job, set(skills.keys()), job_titles, level
            )
            job['_id'] = str(job['_id'])  # Convert ObjectId to string
        
        # Sort by match score
        jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        client.close()
        
        return {
            'jobs': jobs,
            'total_matches': total_matches,
            'page': page,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1
        }
        
    except Exception as e:
        logger.error(f"Error in get_matching_jobs: {e}", exc_info=True)
        raise


def suggest_jobs(resume_text: str, model_name: str = "qwen2.5:3b", limit: int = 10) -> dict:
    """
    Main function to suggest jobs based on resume.
    
    Returns: {matching_jobs, total_matches, extracted_data}
    """
    try:
        # Extract resume data
        resume_data = extract_skills_and_experience(resume_text, model_name)
        logger.info(f"Extracted: {json.dumps(resume_data, indent=2)}")
        
        # Get matching jobs
        result = get_matching_jobs(resume_data, limit=limit)
        
        # Format response
        formatted_jobs = []
        for job in result['jobs']:
            formatted_jobs.append({
                "id": job.get("_id", ""),
                "title": job.get("title", "No Title"),
                "company": job.get("companyName", job.get("company", "N/A")),
                "location": job.get("location", "N/A"),
                "match_percentage": int(job.get("match_score", 0) * 100),
                "skills": job.get("skills", []),
                "experience_level": job.get("experience_level", "N/A")
            })
        
        return {
            "matching_jobs": formatted_jobs,
            "total_matches": result['total_matches'],
            "extracted_data": {
                "job_titles": resume_data.get("job_titles", []),
                "skills": [s['name'] if isinstance(s, dict) else s for s in resume_data.get("technical_skills", [])],
                "experience_level": resume_data.get("experience_level")
            }
        }
        
    except Exception as e:
        logger.error(f"Error in suggest_jobs: {e}", exc_info=True)
        return {
            "error": str(e),
            "matching_jobs": [],
            "total_matches": 0
        }