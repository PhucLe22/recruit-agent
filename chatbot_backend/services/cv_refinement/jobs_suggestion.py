import os
import json
import re
import logging
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from utils.llm_utils import get_ollama_response
from db.database import get_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def extract_skills_and_experience(resume_text: str, model_name: str = "llama2") -> dict:
    """
    Extract skills and experience from resume text using Ollama.
    """
    # Create a prompt for Ollama
    prompt = textwrap.dedent(f"""
        Analyze this resume and extract the following information in JSON format:
        
        1. technical_skills: List of technical skills with years of experience
        2. job_titles: List of job titles/roles
        3. industries: List of industries/domains
        4. experience_level: One of ["internship", "entry", "mid", "senior", "executive"]
        5. education_level: Highest education level
        
        Format your response as a valid JSON object with the fields above.
        
        Resume:
        {resume_text}
        
        JSON Output:
    """)
    
    try:
        # Get response from Ollama
        response = get_ollama_response(prompt=prompt, model=model_name)
        
        # Extract JSON from response
        result_text = response.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].strip()
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()
        
        # Parse the JSON response
        result = json.loads(result_text)
        
        # Ensure all required fields exist
        if not isinstance(result, dict):
            raise ValueError("Invalid response format from LLM")
            
        # Set default values for missing fields
        result.setdefault("technical_skills", [])
        result.setdefault("job_titles", [])
        result.setdefault("industries", [])
        result.setdefault("experience_level", "mid")
        result.setdefault("education_level", "")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_skills_and_experience: {str(e)}")
        return {
            "technical_skills": [],
            "job_titles": [],
            "industries": [],
            "experience_level": "mid",
            "education_level": ""
        }

def get_matching_jobs(
    resume_data: dict, 
    limit: int = 10,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    salary_range: Optional[tuple] = None,
    job_type: Optional[str] = None,
    remote: Optional[bool] = None,
    sort_by: str = 'relevance',
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Advanced job matching function with flexible criteria and pagination.
    
    Args:
        resume_data: Dictionary containing parsed resume data
        limit: Maximum number of jobs to return per page
        location: Location filter (city, state, or country)
        experience_level: Filter by experience level (entry, mid, senior, executive)
        salary_range: Tuple of (min_salary, max_salary)
        job_type: Type of job (full-time, part-time, contract, etc.)
        remote: Whether to filter for remote jobs
        sort_by: Field to sort by (relevance, date_posted, salary, etc.)
        page: Page number for pagination
        page_size: Number of results per page
        
    Returns:
        Dictionary containing:
        - jobs: List of matching jobs with match scores and details
        - total_matches: Total number of matching jobs
        - page: Current page number
        - total_pages: Total number of pages
        - has_next: Whether there are more pages
        - has_previous: Whether there are previous pages
    """
    try:
        logger.info("Starting advanced job matching process...")
        start_time = datetime.now()
        
        # Extract and process resume data
        skills = set()
        if 'technical_skills' in resume_data:
            if isinstance(resume_data['technical_skills'], list):
                skills = {
                    skill['name'].lower().strip(): skill.get('experience_years', 0) 
                    for skill in resume_data['technical_skills'] 
                    if skill and isinstance(skill, dict) and 'name' in skill
                }
        
        job_titles = []
        if 'job_titles' in resume_data and isinstance(resume_data['job_titles'], list):
            job_titles = [str(title).lower().strip() for title in resume_data['job_titles'] if title]
        
        level = str(resume_data.get('level', '')).lower()
        industries = set()
        if 'industries' in resume_data and isinstance(resume_data['industries'], list):
            industries = {str(industry).lower().strip() for industry in resume_data['industries'] if industry}
        
        education = str(resume_data.get('education_level', '')).lower()
        
        logger.info(f"Processing resume with {len(skills)} skills, {len(job_titles)} job titles, and {len(industries)} industries")
        
        if not skills and not job_titles and not industries:
            logger.warning("No skills, job titles, or industries found in resume")
            return {
                'jobs': [],
                'total_matches': 0,
                'page': page,
                'total_pages': 0,
                'has_next': False,
                'has_previous': False
            }
        
        # Connect to MongoDB with connection pooling and timeout
        mongo_uri = os.getenv("MONGO_ATLAS_URI")
        print(mongo_uri.limit(10))
        if not mongo_uri:
            raise ValueError("MONGO_ATLAS_URI not found in environment variables")
        
        client = MongoClient(
            mongo_uri,
            maxPoolSize=100,  # Increased pool size for better performance
            minPoolSize=10,
            maxIdleTimeMS=30000,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        try:
            db = client["CVProject"]
            jobs_collection = db["jobs"]
            
            # Create text index if it doesn't exist
            if 'job_title_text' not in jobs_collection.index_information():
                jobs_collection.create_index([("title", "text"), ("description", "text")], name="job_title_text")
            
            # Build the query
            query = {}
            
            # Text search for job titles and descriptions
            if job_titles or skills:
                search_terms = []
                if job_titles:
                    search_terms.extend(job_titles)
                if skills:
                    search_terms.extend(skills.keys())
                
                if search_terms:
                    query['$text'] = {'$search': ' '.join(f'"{term}"' for term in search_terms[:10])}
            
            # Location filter
            if location:
                location_regex = re.compile(re.escape(location), re.IGNORECASE)
                query['$or'] = [
                    {'location': location_regex},
                    {'city': location_regex},
                    {'state': location_regex},
                    {'country': location_regex}
                ]
            
            # Experience level filter
            if experience_level:
                query['experience_level'] = {'$regex': f'^{re.escape(experience_level)}$', '$options': 'i'}
            
            # Salary range filter
            if salary_range and len(salary_range) == 2:
                min_salary, max_salary = salary_range
                if min_salary is not None and max_salary is not None:
                    query['$or'] = [
                        {'min_salary': {'$gte': min_salary}, 'max_salary': {'$lte': max_salary}},
                        {'min_salary': {'$lte': max_salary}, 'max_salary': {'$gte': min_salary}},
                        {'min_salary': {'$exists': False}, 'max_salary': {'$lte': max_salary}},
                        {'max_salary': {'$exists': False}, 'min_salary': {'$gte': min_salary}}
                    ]
            
            # Job type filter
            if job_type:
                query['job_type'] = {'$regex': f'^{re.escape(job_type)}$', '$options': 'i'}
            
            # Remote filter
            if remote is not None:
                query['is_remote'] = bool(remote)
            
            # Industry filter
            if industries:
                industry_regex = '|'.join(re.escape(industry) for industry in industries)
                query['$or'] = [
                    {'industry': {'$regex': industry_regex, '$options': 'i'}},
                    {'industries': {'$in': list(industries)}}
                ]
            
            # Calculate skip for pagination
            skip = (page - 1) * page_size
            
            # Determine sort order
            sort_criteria = []
            if sort_by == 'relevance' and ('$text' in query or job_titles or skills):
                sort_criteria.append(('score', {'$meta': 'textScore'}))
            elif sort_by == 'date_posted':
                sort_criteria.append(('posted_date', -1))
            elif sort_by == 'salary':
                sort_criteria.append(('max_salary', -1))
            
            # Default sort by relevance if no specific sort is provided
            if not sort_criteria and ('$text' in query or job_titles or skills):
                sort_criteria.append(('score', {'$meta': 'textScore'}))
            
            # Execute the query with projection
            projection = {
                'title': 1,
                'company': 1,
                'location': 1,
                'description': 1,
                'requirements': 1,
                'responsibilities': 1,
                'skills': 1,
                'experience_level': 1,
                'job_type': 1,
                'is_remote': 1,
                'min_salary': 1,
                'max_salary': 1,
                'currency': 1,
                'posted_date': 1,
                'apply_url': 1,
                'company_logo': 1,
                'score': {'$meta': 'textScore'} if sort_criteria and sort_criteria[0][0] == 'score' else 0
            }
            
            # Get total count for pagination
            total_matches = jobs_collection.count_documents(query)
            total_pages = (total_matches + page_size - 1) // page_size
            
            # Execute the query with pagination
            cursor = jobs_collection.find(query, projection)
            
            # Apply sorting if specified
            if sort_criteria:
                cursor = cursor.sort(sort_criteria)
            
            # Apply pagination
            cursor = cursor.skip(skip).limit(page_size)
            
            # Get the results
            jobs = list(cursor)
            
            # Calculate match scores for each job
            for job in jobs:
                job['match_score'] = calculate_job_match_score(job, skills, job_titles, industries, level, education)
            
            # Sort by match score if not already sorted
            if not sort_criteria or sort_criteria[0][0] != 'score':
                jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            # Format the results
            result = {
                'jobs': jobs,
                'total_matches': total_matches,
                'page': page,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_previous': page > 1,
                'query_metrics': {
                    'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'results_count': len(jobs),
                    'filters_applied': {
                        'location': bool(location),
                        'experience_level': bool(experience_level),
                        'salary_range': bool(salary_range),
                        'job_type': bool(job_type),
                        'remote': remote is not None
                    }
                }
            }
            
            return result
            
        finally:
            # Ensure the client is closed properly
            client.close()
    
    except Exception as e:
        logger.error(f"Error in get_matching_jobs: {str(e)}", exc_info=True)
        raise

def calculate_job_match_score(
    job: Dict[str, Any],
    skills: Dict[str, float],
    job_titles: List[str],
    industries: Set[str],
    experience_level: str,
    education: str
) -> float:
    """
    Calculate a match score between a job and a candidate's profile.
    
    Args:
        job: Job document from the database
        skills: Dictionary of skills with years of experience
        job_titles: List of job titles from the resume
        industries: Set of industries from the resume
        experience_level: Experience level from the resume
        education: Education level from the resume
        
    Returns:
        Float score between 0 and 1 representing the match quality
    """
    score = 0.0
    max_possible_score = 0.0
    
    # Weights for different factors (sum should be 1.0)
    weights = {
        'title': 0.25,
        'skills': 0.35,
        'experience': 0.15,
        'education': 0.1,
        'industry': 0.1,
        'location': 0.05
    }
    
    # 1. Title matching
    if job_titles and 'title' in job:
        job_title = job['title'].lower()
        for title in job_titles:
            if title.lower() in job_title or job_title in title.lower():
                score += weights['title']
                break
    max_possible_score += weights['title']
    
    # 2. Skills matching
    if skills and ('skills' in job or 'requirements' in job or 'required_skills' in job):
        job_skills = set()
        print('job_skills: ', job_skills)
        
        # Extract skills from job requirements
        if 'skills' in job:
            if isinstance(job['skills'], list):
                job_skills.update(skill.lower() for skill in job['skills'] if isinstance(skill, str))
            elif isinstance(job['skills'], str):
                job_skills.update(skill.strip().lower() for skill in job['skills'].split(','))
        
        # Check for alternative skill fields
        for field in ['requirements', 'required_skills', 'preferred_skills']:
            if field in job and isinstance(job[field], str):
                requirements_text = job[field].lower()
                # Split by common separators and clean up
                for word in re.split(r'[,\n•;]', requirements_text):
                    word = word.strip().lower()
                    if word and len(word) > 2:  # Skip very short words
                        job_skills.add(word)
        
        # Also check for skills in the job description
        if 'description' in job and isinstance(job['description'], str):
            desc_text = job['description'].lower()
            for skill in skills:
                if skill.lower() in desc_text and skill.lower() not in job_skills:
                    job_skills.add(skill.lower())
        
        # Calculate skill match score with fuzzy matching
        if job_skills:
            matched_skills = set()
            resume_skills_lower = {k.lower(): v for k, v in skills.items()}
            
            # First pass: exact matches
            for job_skill in job_skills:
                if job_skill in resume_skills_lower:
                    matched_skills.add(job_skill)
            
            # Second pass: partial matches (if no exact matches found)
            if not matched_skills:
                for job_skill in job_skills:
                    for resume_skill in resume_skills_lower:
                        if job_skill in resume_skill or resume_skill in job_skill:
                            matched_skills.add(job_skill)
                            break
            
            # Calculate score based on matched skills
            if matched_skills:
                skill_score = sum(resume_skills_lower.get(skill, 0) for skill in matched_skills)
                max_skill_score = sum(resume_skills_lower.values())
                
                if max_skill_score > 0:
                    normalized_skill_score = skill_score / max_skill_score
                    score += normalized_skill_score * weights['skills']
                    
                    # Log the matching for debugging
                    logger.info(f"Matched skills for job {job.get('title', 'Unknown')}:")
                    logger.info(f"- Required: {job_skills}")
                    logger.info(f"- Matched: {matched_skills}")
                    logger.info(f"- All resume skills: {list(resume_skills_lower.keys())}")
    
    max_possible_score += weights['skills']
    
    # 3. Experience level matching
    if experience_level and 'experience_level' in job:
        job_exp_level = job['experience_level'].lower()
        exp_levels = ['entry', 'mid', 'senior', 'executive']
        try:
            candidate_level = exp_levels.index(experience_level.lower())
            job_level = exp_levels.index(job_exp_level)
            if candidate_level >= job_level:  # Candidate meets or exceeds required level
                score += weights['experience']
            else:
                score += weights['experience'] * (candidate_level / max(job_level, 1))
            max_possible_score += weights['experience']
        except ValueError:
            pass  # One of the levels wasn't in our list
    
    # 4. Education matching (10% weight)
    if education and 'education_required' in job:
        education_levels = {
            'high school': 1,
            'associate': 2,
            "bachelor's": 3,
            "master's": 4,
            'phd': 5
        }
        try:
            candidate_edu = education_levels[education.lower()]
            job_edu = education_levels[job['education_required'].lower()]
            if candidate_edu >= job_edu:  # Candidate meets or exceeds required education
                score += 0.1
            else:
                score += 0.1 * (candidate_edu / max(job_edu, 1))
            max_possible_score += 0.1
        except (KeyError, AttributeError):
            pass
    
    # 5. Industry matching (5% weight)
    if industries and 'industries' in job:
        job_industries = set(industry.lower() for industry in job['industries'])
        if industries.intersection(job_industries):
            score += 0.05
            max_possible_score += 0.05
    
    # Normalize the score to be between 0 and 1
    normalized_score = score / max_possible_score if max_possible_score > 0 else 0
    
    # Apply a sigmoid function to make the score more discriminative
    # This will push scores away from 0.5 (higher scores get closer to 1, lower scores closer to 0)
    import math
    if normalized_score > 0:
        normalized_score = 1 / (1 + math.exp(-10 * (normalized_score - 0.5)))
    
    return round(normalized_score, 4)

def get_matching_jobs(
    resume_data: dict, 
    limit: int = 10,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    salary_range: Optional[tuple] = None,
    job_type: Optional[str] = None,
    remote: Optional[bool] = None,
    sort_by: str = 'relevance',
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Advanced job matching function with flexible criteria and pagination.
    
    Args:
        resume_data: Dictionary containing parsed resume data
        limit: Maximum number of jobs to return per page
        location: Location filter (city, state, or country)
        experience_level: Filter by experience level (entry, mid, senior, executive)
        salary_range: Tuple of (min_salary, max_salary)
        job_type: Type of job (full-time, part-time, contract, etc.)
        remote: Whether to filter for remote jobs
        sort_by: Field to sort by (relevance, date_posted, salary, etc.)
        page: Page number for pagination
        page_size: Number of results per page
        
    Returns:
        Dictionary containing:
        - jobs: List of matching jobs with match scores and details
        - total_matches: Total number of matching jobs
        - page: Current page number
        - total_pages: Total number of pages
        - has_next: Whether there are more pages
        - has_previous: Whether there are previous pages
    """
    try:
        logger.info("Starting advanced job matching process...")
        start_time = datetime.now()
        
        # Extract and process resume data
        skills = {}
        if 'technical_skills' in resume_data:
            if isinstance(resume_data['technical_skills'], list):
                skills = {
                    skill['name'].lower().strip(): skill.get('experience_years', 0) 
                    for skill in resume_data['technical_skills'] 
                    if 'name' in skill
                }
        
        job_titles = []
        if 'job_titles' in resume_data and isinstance(resume_data['job_titles'], list):
            job_titles = [title.lower().strip() for title in resume_data['job_titles'] if title]
        
        # Create a regex pattern for title matching
        title_pattern = "|\\.\\.\\.\\b".join([re.escape(title) for title in job_titles]) if job_titles else ""
        
        # Build the aggregation pipeline
        pipeline = [
            # First, find jobs that match either title or skills
            {
                "$match": {
                    "$and": [
                        # Must match at least one of these conditions
                        {"$or": [
                            # Match job titles (case insensitive) only if we have job titles
                            *([{"title": {"$regex": title_pattern, "$options": "i"}}] if job_titles else []),
                            # Or match any of the required skills
                            {"required_skills": {"$in": list(skills)}},
                            # Or match any of the preferred skills
                            {"preferred_skills": {"$in": list(skills)}}
                        ]}
                    ]
                }
            },
            # Add fields for scoring
            {
                "$addFields": {
                    # Count matching required skills
                    "matched_required_skills": {
                        "$size": {
                            "$setIntersection": [
                                {"$ifNull": ["$required_skills", []]},
                                list(skills)
                            ]
                        }
                    },
                    # Count matching preferred skills
                    "matched_preferred_skills": {
                        "$size": {
                            "$setIntersection": [
                                {"$ifNull": ["$preferred_skills", []]},
                                list(skills)
                            ]
                        }
                    },
                    # Check title match
                    "title_match": {
                        "$cond": [
                            {"$gt": [
                                {"$size": {
                                    "$filter": {
                                        "input": job_titles,
                                        "as": "title",
                                        "cond": {
                                            "$regexMatch": {
                                                "input": "$title",
                                                "regex": "$$title",
                                                "options": "i"
                                            }
                                        }
                                    }
                                }}, 0]
                            },
                            1,
                            0
                        ]
                    },
                    # Check level match
                    "level_match": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$eq": ["$experience_level", experience_level]},
                                    {"$not": ["$experience_level"]},
                                    {"$eq": ["$experience_level", ""]},
                                    {"$eq": ["$experience_level", None]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            },
            # Calculate scores
            {
                "$addFields": {
                    # Base score from required skills (50% weight)
                    "required_skills_score": {
                        "$multiply": [
                            {
                                "$cond": [
                                    {"$gt": [{"$size": {"$ifNull": ["$required_skills", []]}}, 0]},
                                    {"$divide": ["$matched_required_skills", {"$size": "$required_skills"}]},
                                    0
                                ]
                            },
                            0.5
                        ]
                    },
                    # Bonus for preferred skills (20% weight)
                    "preferred_skills_score": {
                        "$multiply": [
                            {
                                "$cond": [
                                    {"$gt": [{"$size": {"$ifNull": ["$preferred_skills", []]}}, 0]},
                                    {"$divide": ["$matched_preferred_skills", {"$size": "$preferred_skills"}]},
                                    0
                                ]
                            },
                            0.2
                        ]
                    },
                    # Title match bonus (15% weight)
                    "title_match_score": {"$multiply": ["$title_match", 0.15]},
                    # Level match bonus (10% weight)
                    "level_match_score": {"$multiply": ["$level_match", 0.1]}
                }
            },
            # Calculate total score (sum of all scores, max 1.0)
            {
                "$addFields": {
                    "match_score": {
                        "$min": [
                            {"$add": [
                                "$required_skills_score",
                                "$preferred_skills_score",
                                "$title_match_score",
                                "$level_match_score"
                            ]},
                            1.0  # Cap at 1.0
                        ]
                    },
                    # Calculate match percentage for display
                    "match_percentage": {
                        "$multiply": [
                            {"$min": [
                                {"$add": [
                                    "$required_skills_score",
                                    "$preferred_skills_score",
                                    "$title_match_score",
                                    "$level_match_score"
                                ]},
                                1.0
                            ]},
                            100
                        ]
                    }
                }
            },
            # Filter out jobs with no matching skills
            {
                "$match": {
                    "$or": [
                        {"matched_required_skills": {"$gt": 0}},
                        {"matched_preferred_skills": {"$gt": 0}}
                    ]
                }
            },
            # Sort by match score (descending)
            {"$sort": {"match_score": -1}},
            # Limit results
            {"$limit": limit},
            # Project final fields
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "company": {"$ifNull": ["$companyName", "$company"]},
                    "location": 1,
                    "experience_level": 1,
                    "required_skills": 1,
                    "preferred_skills": 1,
                    "match_score": 1,
                    "match_percentage": 1,
                    "matched_skills": {
                        "$setUnion": [
                            {"$setIntersection": ["$required_skills", list(skills)]},
                            {"$setIntersection": ["$preferred_skills", list(skills)]}
                        ]
                    },
                    "score_breakdown": {
                        "required_skills": {"$round": [{"$multiply": ["$required_skills_score", 100]}, 1]},
                        "preferred_skills": {"$round": [{"$multiply": ["$preferred_skills_score", 100]}, 1]},
                        "title_match": {"$round": [{"$multiply": ["$title_match_score", 100]}, 1]},
                        "level_match": {"$round": [{"$multiply": ["$level_match_score", 100]}, 1]}
                    }
                }
            }
        ]
        
        # Execute the pipeline
        logger.info("Executing MongoDB aggregation pipeline...")
        matching_jobs = list(jobs_collection.aggregate(pipeline))
        
        # Filter out jobs that don't have any matching skills
        filtered_jobs = []
        for job in matching_jobs:
            # Calculate skill matches
            req_skills_matched = len(set(job.get('required_skills', [])).intersection(skills))
            pref_skills_matched = len(set(job.get('preferred_skills', [])).intersection(skills))
            
            # Only include jobs with relevant matches
            if req_skills_matched > 0 or pref_skills_matched >= 2:
                filtered_jobs.append(job)
        
        logger.info(f"Found {len(filtered_jobs)} relevant jobs out of {len(matching_jobs)} initial matches")
        matching_jobs = filtered_jobs
        
        # return matching_jobs
        
        # Define weights for different matching criteria
        WEIGHTS = {
            'job_title': 1.5,           # Increased weight for job title match
            'company': 0.3,             # Reduced weight for company
            'experience_level': 0.4,    # Slightly increased weight for experience level
            'required_skills': 0.8,     # Increased weight for required skills
            'preferred_skills': 0.2,    # Preferred skills are a plus
            'industry': 0.1             # Industry match is a plus
        }
        
        # High threshold for job matching (85% match required)
        MIN_SCORE_THRESHOLD = 0.85
        
        # Minimum requirements for job matching
        MIN_REQUIRED_SKILLS_RATIO = 0.7   # At least 70% of required skills must match
        MIN_PREFERRED_SKILLS_RATIO = 0.4  # At least 40% of preferred skills should match
        
        # Minimum absolute number of required skills that must match
        MIN_REQUIRED_SKILLS_COUNT = 3
        
        # Minimum score for title and company to be considered a match
        MIN_TITLE_MATCH_SCORE = 0.7
        MIN_COMPANY_MATCH_SCORE = 0.5
        
        # Minimum number of skills that must match (absolute count)
        MIN_SKILLS_MATCH_COUNT = 3
        
        def extract_company_name(job_title: str) -> str:
            """Extract company name from job title using multiple patterns."""
            # Common patterns for company names in job titles
            patterns = [
                r'at\s+([A-Z][A-Za-z0-9&.\-\s]+)(?:\s+\(|(?:\s+at\s|$))',
                r'@\s*([A-Z][A-Za-z0-9&.\-\s]+)(?:\s*\||$|\s+at\s)',
                r'\b(?:at|@)\s+([A-Z][A-Za-z0-9&.\-\s]+)(?:\s*\||$|\s+at\s)',
                r'\b(?:for|from|by|at)\s+([A-Z][A-Za-z0-9&.\-\s]+)(?:\s*\||$|\s+for\s|\s+at\s)'
            ]
            
            # Try patterns in order
            for pattern in patterns:
                match = re.search(pattern, job_title, re.IGNORECASE)
                if match:
                    company = match.group(1).strip()
                    # Clean up common suffixes
                    company = re.sub(r'\s*(?:LLC|Inc|Ltd|Corp|Pte\.?|Lt\.?|Co\.?|GmbH)\b', '', company, flags=re.IGNORECASE)
                    return company.strip()
            
            # If no pattern matched, try to extract company-like words
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', job_title)
            if len(words) > 1:
                # Return the last word that looks like a company name
                for word in reversed(words):
                    if len(word) > 2 and word.lower() not in ['for', 'and', 'the', 'with', 'using']:
                        return word
            
            return ""
        
        # Extract and process data from resume
        skills = set(skill['name'].lower().strip() for skill in resume_data.get('technical_skills', []))
        level = resume_data.get('level', '').lower()
        industries = set(industry.lower() for industry in resume_data.get('industries', []))
        job_titles = set(title.lower() for title in resume_data.get('job_titles', []))
        
        # Extract company names from job titles
        companies = set()
        for title in job_titles:
            company = extract_company_name(title)
            if company:
                companies.add(company.lower())
        
        # If no companies found in titles, try to extract from work experience
        if not companies and 'work_experience' in resume_data:
            for exp in resume_data['work_experience']:
                if 'company' in exp:
                    companies.add(exp['company'].lower())
        
        logger.info(f"Extracted companies from resume: {companies}")
        logger.info(f"Job titles from resume: {job_titles}")
        logger.info(f"Experience level: {level}")
        logger.info(f"Skills: {list(skills)[:5]}... (total: {len(skills)})")
        
        if not skills and not job_titles and not industries:
            logger.warning("No relevant data found in resume for job matching")
            return []
        
        # Connect to MongoDB
        mongo_uri = os.getenv("MONGO_ATLAS_URI")
        if not mongo_uri:
            raise ValueError("MONGO_ATLAS_URI not found in environment variables")
            
        client = MongoClient(mongo_uri)
        db = client["CVProject"]
        jobs_collection = db["jobs"]
        
        # First, get all jobs that match the title or required skills
        pipeline = [
            # First phase: Find potential matches based on title or required skills
            {
                "$match": {
                    "$or": [
                        # Match job title exactly or partially (higher priority)
                        {
                            "$or": [
                                {"title": {"$in": list(job_titles)}},
                                {"title": {"$regex": "|\\.".join(re.escape(t) for t in job_titles), "$options": "i"}}
                            ]
                        },
                        # Or match required skills (lower priority)
                        {
                            "required_skills": {
                                "$in": list(skills)
                            }
                        }
                    ]
                }
            },
            # Add fields for title and company matching scores
            {
                "$addFields": {
                    # Calculate title match score (0-1)
                    "title_match_score": {
                        "$max": [
                            {
                                "$cond": [
                                    {"$in": [{"$toLower": "$title"}, [t.lower() for t in job_titles]]},
                                    1.0,
                                    0.0
                                ]
                            },
                            {
                                "$max": [
                                    {
                                        "$reduce": {
                                            "input": job_titles,
                                            "initialValue": 0.0,
                                            "in": {
                                                "$max": [
                                                    "$$value",
                                                    {
                                                        "$cond": [
                                                            {"$regexMatch": {
                                                                "input": "$title",
                                                                "regex": f"{re.escape('$$this')}",
                                                                "options": "i"
                                                            }},
                                                            0.8,  # Partial match score
                                                            0.0
                                                        ]
                                                    }
                                                ]
                                            }
                                        }
                                    },
                                    0.0
                                ]
                            }
                        ]
                    },
                    # Calculate company match score (0-1)
                    "company_match_score": {
                        "$max": [
                            {
                                "$cond": [
                                    {
                                        "$or": [
                                            {"$in": [{"$toLower": "$company"}, [c.lower() for c in companies]]},
                                            {"$in": [{"$toLower": "$companyName"}, [c.lower() for c in companies]]}
                                        ]
                                    },
                                    1.0,
                                    0.0
                                ]
                            },
                            {
                                "$max": [
                                    {
                                        "$reduce": {
                                            "input": companies,
                                            "initialValue": 0.0,
                                            "in": {
                                                "$max": [
                                                    "$$value",
                                                    {
                                                        "$cond": [
                                                            {
                                                                "$or": [
                                                                    {"$regexMatch": {
                                                                        "input": "$company",
                                                                        "regex": f"{re.escape('$$this')}",
                                                                        "options": "i"
                                                                    }},
                                                                    {"$regexMatch": {
                                                                        "input": "$companyName",
                                                                        "regex": f"{re.escape('$$this')}",
                                                                        "options": "i"
                                                                    }}
                                                                ]
                                                            },
                                                            0.8,  # Partial match score
                                                            0.0
                                                        ]
                                                    }
                                                ]
                                            }
                                        }
                                    },
                                    0.0
                                ]
                            }
                        ]
                    }
                }
            },
            # Second phase: Filter by skills and experience level
            {
                "$match": {
                    "$and": [
                        # Must have at least one strong match (title or skills)
                        {
                            "$or": [
                                # Either have a strong title match
                                {"title_match_score": {"$gte": MIN_TITLE_MATCH_SCORE}},
                                # Or have enough matching skills
                                {
                                    "$expr": {
                                        "$gte": [
                                            {
                                                "$size": {
                                                    "$setIntersection": [
                                                        {"$ifNull": ["$required_skills", []]},
                                                        list(skills)
                                                    ]
                                                }
                                            },
                                            MIN_SKILLS_MATCH_COUNT
                                        ]
                                    }
                                }
                            ]
                        },
                        # Must match experience level (if specified in resume)
                        {"$or": [
                            {"experience_level": level},
                            {"experience_level": {"$exists": False}},
                            {"experience_level": ""},
                            {"$expr": {"$eq": ["$experience_level", None]}}
                        ]},
                        # Must have at least some required skills match
                        {
                            "$expr": {
                                "$gte": [
                                    {"$size": {"$setIntersection": ["$required_skills", list(skills)]}},
                                    MIN_REQUIRED_SKILLS_COUNT
                                ]
                            }
                        }
                    ]
                }
            },
            # Calculate match scores
            {
                "$addFields": {
                    # Calculate skill match scores
                    "required_skills_match": {
                        "$size": {
                            "$setIntersection": [
                                {"$ifNull": ["$required_skills", []]},
                                list(skills)
                            ]
                        }
                    },
                    "preferred_skills_match": {
                        "$size": {
                            "$setIntersection": [
                                {"$ifNull": ["$preferred_skills", []]},
                                list(skills)
                            ]
                        }
                    },
                    # Check level match
                    "level_match": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$eq": ["$experience_level", level]},
                                    {"$not": ["$experience_level"]}  # If job doesn't specify level, consider it a match
                                ]
                            },
                            1,
                            0
                        ]
                    },
                    # Check industry match
                    "industry_match": {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$size": {
                                            "$setIntersection": [
                                                {"$ifNull": ["$industries", []]},
                                                list(industries)
                                            ]
                                        }
                                    },
                                    0
                                ]
                            },
                            1,
                            0
                        ]
                    },
                    # Check job title match
                    "title_match": {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$size": {
                                            "$filter": {
                                                "input": list(job_titles),
                                                "as": "title",
                                                "cond": {
                                                    "$regexMatch": {
                                                        "input": "$title",
                                                        "regex": "$$title",
                                                        "options": "i"
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    0
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            },
            # Calculate skill match ratios
            {
                "$addFields": {
                    # Calculate required skills ratio (0-1)
                    "required_skills_ratio": {
                        "$cond": [
                            {"$gt": [{"$size": {"$ifNull": ["$required_skills", []]}}, 0]},
                            {"$divide": ["$required_skills_match", {"$size": "$required_skills"}]},
                            0
                        ]
                    },
                    # Calculate preferred skills ratio (0-1)
                    "preferred_skills_ratio": {
                        "$cond": [
                            {"$gt": [{"$size": {"$ifNull": ["$preferred_skills", []]}}, 0]},
                            {"$divide": ["$preferred_skills_match", {"$size": "$preferred_skills"}]},
                            0
                        ]
                    },
                    # Industry match (1 if any industry matches, 0 otherwise)
                    "industry_match": {
                        "$cond": [
                            {
                                "$gt": [
                                    {"$size": {
                                        "$setIntersection": [
                                            {"$ifNull": ["$industries", []]},
                                            list(industries)
                                        ]
                                    }},
                                    0
                                ]
                            },
                            1.0,
                            0.0
                        ]
                    }
                }
            },
            # Calculate final weighted score with job title and company having highest priority
            {
                "$addFields": {
                    "match_score": {
                        "$add": [
                            # Job title match has highest weight
                            {"$multiply": ["$title_match_score", WEIGHTS['job_title']]},
                            # Company match is also very important
                            {"$multiply": ["$company_match_score", WEIGHTS['company']]},
                            # Experience level is important
                            {"$multiply": ["$level_match", WEIGHTS['experience_level']]},
                            # Required skills are still important but slightly less than title/company
                            {"$multiply": ["$required_skills_ratio", WEIGHTS['required_skills']]},
                            # Preferred skills and industry are nice-to-haves
                            {"$multiply": ["$preferred_skills_ratio", WEIGHTS['preferred_skills']]},
                            {"$multiply": ["$industry_match", WEIGHTS['industry']]}
                        ]
                    },
                    # Add detailed scoring information for debugging
                    "scoring_details": {
                        "title_score": {"$multiply": ["$title_match_score", WEIGHTS['job_title']]},
                        "company_score": {"$multiply": ["$company_match_score", WEIGHTS['company']]},
                        "level_score": {"$multiply": ["$level_match", WEIGHTS['experience_level']]},
                        "required_skills_score": {"$multiply": ["$required_skills_ratio", WEIGHTS['required_skills']]},
                        "preferred_skills_score": {"$multiply": ["$preferred_skills_ratio", WEIGHTS['preferred_skills']]},
                        "industry_score": {"$multiply": ["$industry_match", WEIGHTS['industry']]}
                    }
                }
            },
            # Add match percentage and detailed scoring info
            {
                "$addFields": {
                    "match_percentage": {
                        "$multiply": ["$match_score", 100]
                    },
                    "score_breakdown": {
                        "job_title": {"$multiply": ["$title_match_score", WEIGHTS['job_title']]},
                        "company": {"$multiply": ["$company_match_score", WEIGHTS['company']]},
                        "experience_level": {"$multiply": ["$level_match", WEIGHTS['experience_level']]},
                        "required_skills": {"$multiply": ["$required_skills_ratio", WEIGHTS['required_skills']]},
                        "preferred_skills": {"$multiply": ["$preferred_skills_ratio", WEIGHTS['preferred_skills']]},
                        "industry": {"$multiply": ["$industry_match", WEIGHTS['industry']]}
                    }
                }
            },
            # Filter out jobs that don't meet minimum criteria
            {
                "$match": {
                    "$expr": {
                        "$and": [
                            # Must meet minimum score threshold (90%+)
                            {"$gte": ["$match_score", MIN_SCORE_THRESHOLD]},
                            # Must have enough matching skills
                            {
                                "$expr": {
                                    "$gte": [
                                        {"$size": {"$setIntersection": ["$required_skills", list(skills)]}},
                                        MIN_REQUIRED_SKILLS_COUNT
                                    ]
                                }
                            },
                            # Must have either strong title match OR strong skills match
                            {
                                "$or": [
                                    # Strong title match with some skills
                                    {
                                        "$and": [
                                            {"$gte": ["$title_match_score", MIN_TITLE_MATCH_SCORE]},
                                            {"$gte": ["$required_skills_ratio", 0.5]}  # At least 50% skills match
                                        ]
                                    },
                                    # OR very strong skills match
                                    {
                                        "$and": [
                                            {"$gte": ["$required_skills_ratio", 0.8]},  # 80%+ skills match
                                            {"$gte": ["$level_match", 0.7]}  # And good level match
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            # Sort by match score and other important factors
            {"$sort": {
                "match_score": -1,                  # Highest match score first
                "title_match_score": -1,             # Then by job title match strength
                "company_match_score": -1,           # Then by company match
                "level_match": -1,                   # Then by experience level match
                "required_skills_ratio": -1,         # Then by ratio of required skills matched
                "preferred_skills_ratio": -1,        # Then by preferred skills
                "industry_match": -1                 # Finally by industry match
            }},
            # Limit results
            {"$limit": limit},
            # Project final fields
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "companyName": 1,
                    "company": 1,  # Some jobs might use different field names
                    "location": 1,
                    "experience_level": 1,
                    "required_skills": 1,
                    "preferred_skills": 1,
                    "industries": 1,
                    "title_match_score": 1,
                    "company_match_score": 1,
                    "required_skills_ratio": 1,
                    "scoring_details": 1,
                    "applyLink": 1,
                    "job_link": 1,
                    "match_score": 1,
                    "match_percentage": 1,
                    "score_breakdown": 1,
                    "title_match_score": 1,
                    "company_match_score": 1,
                    "required_skills_ratio": 1,
                    "preferred_skills_ratio": 1,
                    "required_skills_match": 1,
                    "preferred_skills_match": 1,
                    "level_match": 1,
                    "industry_match": 1
                }
            }
        ]
        
        # Execute the aggregation
        jobs = list(jobs_collection.aggregate(pipeline))
        
        # Calculate max possible score for normalization
        max_possible_score = (
            WEIGHTS['required_skills'] * len(skills) +
            WEIGHTS['preferred_skills'] * len(skills) +
            WEIGHTS['experience_level'] * 1 +
            WEIGHTS['industry'] * 1 +
            WEIGHTS['job_title'] * 1
        )
        
        # Format the results with detailed matching information
        results = []
        for job in jobs:
            # Calculate match percentage with a boost factor
            raw_score = job.get("match_score", 0)
            # Add a boost factor to increase the percentage
            boost_factor = 1.2  # 20% boost to the final score
            normalized_score = (raw_score / max_possible_score) if max_possible_score > 0 else 0
            # Apply a square root to the normalized score to make it more forgiving
            # This will give higher percentages for partial matches
            adjusted_score = (normalized_score ** 0.9) * boost_factor
            # Ensure we don't exceed 100%
            match_percentage = min(100, int(adjusted_score * 100))
            
            # Get matched skills
            required_skills_matched = set(skill.lower() for skill in job.get('required_skills', []) if skill.lower() in skills)
            preferred_skills_matched = set(skill.lower() for skill in job.get('preferred_skills', []) if skill.lower() in skills)
            all_matched_skills = list(required_skills_matched.union(preferred_skills_matched))
            
            # Get match details
            level_matched = job.get("level_match", 0) == 1
            industry_matched = job.get("industry_match", 0) == 1
            title_matched = job.get("title_match", 0) == 1
            
            # Calculate score breakdown
            score_breakdown = {
                "required_skills": {
                    "score": len(required_skills_matched) * WEIGHTS['required_skills'],
                    "matched": list(required_skills_matched),
                    "total": len(job.get('required_skills', [])),
                    "weight": WEIGHTS['required_skills']
                },
                "preferred_skills": {
                    "score": len(preferred_skills_matched) * WEIGHTS['preferred_skills'],
                    "matched": list(preferred_skills_matched),
                    "total": len(job.get('preferred_skills', [])),
                    "weight": WEIGHTS['preferred_skills']
                },
                "level": {
                    "score": WEIGHTS['experience_level'] if level_matched else 0,
                    "matched": level_matched,
                    "job_level": job.get("experience_level", "Not specified"),
                    "resume_level": level,
                    "weight": WEIGHTS['experience_level']
                },
                "industry": {
                    "score": WEIGHTS['industry'] if industry_matched else 0,
                    "matched": industry_matched,
                    "job_industries": job.get("industries", []),
                    "resume_industries": list(industries),
                    "weight": WEIGHTS['industry']
                },
                "job_title": {
                    "score": WEIGHTS['job_title'] if title_matched else 0,
                    "matched": title_matched,
                    "job_title": job.get("title", ""),
                    "resume_titles": list(job_titles),
                    "weight": WEIGHTS['job_title']
                }
            }
            
            # Prepare the result entry
            result = {
                "id": str(job.get("_id", "")),
                "title": job.get("title", ""),
                "company": job.get("companyName", job.get("company", "")),
                "location": job.get("location", ""),
                "experience_level": job.get("experience_level", "Not specified"),
                "industries": job.get("industries", []),
                "required_skills": job.get("required_skills", []),
                "preferred_skills": job.get("preferred_skills", []),
                "matched_skills": all_matched_skills,
                "match_percentage": match_percentage,
                "match_details": score_breakdown,
                "job_link": job.get("applyLink", job.get("job_link", ""))
            }
            
            results.append(result)
            
            # Log detailed matching information for debugging
            logger.info(f"Job match - Title: {result['title']}, Company: {result['company']}")
            logger.info(f"  Match Score: {match_percentage}%")
            logger.info(f"  Matched Skills: {len(all_matched_skills)}/{len(required_skills_matched) + len(preferred_skills_matched)}")
            logger.info(f"  Level Matched: {level_matched} (Job: {result['experience_level']}, Resume: {level})")
            logger.info(f"  Industry Matched: {industry_matched}")
            logger.info(f"  Title Matched: {title_matched}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error querying jobs: {str(e)}")
        return []

from typing import Set, Dict, List, Any, Optional
import re
from collections import defaultdict
import spacy

# Load spaCy model for NLP processing
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

# Skill categories for better organization
SKILL_CATEGORIES = {
    'programming': {
        'languages': {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin',
            'go', 'rust', 'scala', 'r', 'matlab', 'dart', 'perl', 'haskell', 'erlang', 'elixir', 'clojure', 'lua'
        },
        'frontend': {
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'material ui', 'chakra ui', 'styled components',
            'redux', 'mobx', 'graphql', 'apollo', 'webpack', 'vite', 'rollup', 'babel', 'jest', 'mocha', 'cypress',
            'storybook', 'next.js', 'gatsby', 'nuxt.js', 'svelte', 'sveltekit', 'angular', 'vue', 'react', 'jquery'
        },
        'backend': {
            'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'laravel', 'ruby on rails',
            'asp.net', '.net core', 'nest.js', 'koa', 'hapi', 'sails.js', 'adonis.js', 'loopback', 'strapi', 'keystone',
            'graphql', 'grpc', 'rest', 'soap', 'fastify', 'moleculer', 'micro', 'feathers.js', 'sails.js'
        },
        'mobile': {
            'react native', 'flutter', 'xamarin', 'ionic', 'native script', 'swift ui', 'kotlin multiplatform',
            'flutter', 'xcode', 'android studio', 'firebase', 'realm', 'sqlite', 'core data', 'room'
        },
        'ai_ml': {
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'opencv', 'nltk', 'spacy', 'huggingface', 'transformers',
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'dash', 'streamlit', 'gradio', 'mlflow',
            'tensorboard', 'pytorch lightning', 'fastai', 'xgboost', 'lightgbm', 'catboost', 'h2o', 'tensorrt', 'onnx'
        },
        'devops': {
            'docker', 'kubernetes', 'helm', 'terraform', 'ansible', 'puppet', 'chef', 'saltstack', 'vault', 'consul',
            'nomad', 'jenkins', 'github actions', 'gitlab ci/cd', 'circleci', 'travis ci', 'argo cd', 'flux', 'spinnaker',
            'istio', 'linkerd', 'kong', 'nginx', 'apache', 'caddy', 'traefik', 'prometheus', 'grafana', 'loki', 'tempo',
            'thanos', 'victoriametrics', 'datadog', 'new relic', 'sentry', 'logstash', 'fluentd', 'filebeat', 'metricbeat'
        },
        'cloud': {
            'aws', 'amazon web services', 'azure', 'google cloud', 'gcp', 'oracle cloud', 'ibm cloud', 'alibaba cloud',
            'digitalocean', 'linode', 'vultr', 'heroku', 'vercel', 'netlify', 'cloudflare', 'cloudfront', 'cloud run',
            'cloud functions', 'lambda', 'ec2', 'ecs', 'eks', 'fargate', 's3', 'rds', 'dynamodb', 'aurora', 'redshift',
            'bigquery', 'bigtable', 'spanner', 'firestore', 'firebase', 'mongodb atlas', 'cosmos db', 'documentdb'
        },
        'databases': {
            'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server', 'sqlite', 'cassandra', 'dynamodb',
            'firebase', 'elasticsearch', 'mariadb', 'neo4j', 'couchbase', 'couchdb', 'rethinkdb', 'influxdb', 'timescaledb',
            'cockroachdb', 'snowflake', 'bigquery', 'redshift', 'hbase', 'hive', 'presto', 'trino', 'clickhouse', 'snowflake'
        },
        'testing': {
            'jest', 'mocha', 'jasmine', 'karma', 'cypress', 'playwright', 'puppeteer', 'selenium', 'appium', 'detox',
            'junit', 'testng', 'pytest', 'unittest', 'rspec', 'cucumber', 'jbehave', 'testcafe', 'webdriverio', 'cypress',
            'postman', 'newman', 'rest assured', 'karate', 'taiko', 'gauge', 'taiko', 'testcafe', 'playwright', 'cypress'
        },
        'tools': {
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'trello', 'asana', 'clickup', 'notion',
            'slack', 'microsoft teams', 'zoom', 'google meet', 'webex', 'figma', 'sketch', 'adobe xd', 'invision', 'zeplin',
            'docker', 'kubernetes', 'vagrant', 'virtualbox', 'vmware', 'vscode', 'intellij', 'eclipse', 'android studio',
            'xcode', 'postman', 'insomnia', 'swagger', 'openapi', 'graphql playground', 'datagrip', 'dbeaver', 'tableplus'
        },
        'methodologies': {
            'agile', 'scrum', 'kanban', 'lean', 'devops', 'devsecops', 'gitops', 'aiops', 'mlops', 'dataops',
            'tdd', 'bdd', 'atdd', 'ci/cd', 'continuous integration', 'continuous delivery', 'continuous deployment',
            'microservices', 'serverless', 'domain driven design', 'event driven architecture', 'clean architecture',
            'hexagonal architecture', 'onion architecture', 'cqrs', 'event sourcing', 'saga pattern', 'strangler pattern'
        },
        'security': {
            'owasp', 'penetration testing', 'security testing', 'vulnerability assessment', 'threat modeling',
            'encryption', 'tls', 'ssl', 'jwt', 'oauth', 'openid connect', 'saml', 'ldap', 'kerberos', 'rbac', 'abac',
            'zero trust', 'vault', 'keycloak', 'okta', 'auth0', 'cognito', 'firebase auth', 'jwt', 'oauth2', 'openid',
            'sso', 'mfa', '2fa', 'biometric authentication', 'fido2', 'webauthn'
        }
    },
    'soft_skills': {
        'communication', 'teamwork', 'problem-solving', 'critical thinking', 'time management', 'leadership',
        'adaptability', 'creativity', 'emotional intelligence', 'conflict resolution', 'negotiation', 'collaboration',
        'decision making', 'stress management', 'work ethic', 'attention to detail', 'analytical thinking',
        'strategic thinking', 'innovation', 'mentoring', 'coaching', 'presentation', 'public speaking', 'networking',
        'interpersonal skills', 'customer service', 'empathy', 'patience', 'resilience', 'flexibility', 'initiative',
        'accountability', 'dependability', 'professionalism', 'work-life balance', 'cultural awareness', 'diversity',
        'inclusion', 'emotional intelligence', 'active listening', 'feedback', 'delegation', 'motivation', 'persuasion',
        'influence', 'resourcefulness', 'self-motivation', 'self-awareness', 'self-regulation', 'social skills'
    },
    'languages': {
        'english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'korean', 'russian', 'portuguese', 'italian',
        'dutch', 'swedish', 'norwegian', 'danish', 'finnish', 'polish', 'turkish', 'arabic', 'hindi', 'bengali',
        'vietnamese', 'thai', 'indonesian', 'malay', 'tagalog', 'swahili', 'yoruba', 'zulu', 'afrikaans', 'latin',
        'ancient greek', 'sanskrit', 'hebrew', 'tamil', 'telugu', 'kannada', 'malayalam', 'marathi', 'gujarati', 'punjabi'
    },
    'certifications': {
        'aws certified', 'microsoft certified', 'google cloud certified', 'oracle certified', 'cisco certified',
        'comptia', 'project management professional', 'certified scrum master', 'certified scrum product owner',
        'certified information systems security professional', 'certified ethical hacker', 'certified cloud security professional',
        'certified data professional', 'certified information security manager', 'certified information systems auditor',
        'certified information privacy professional', 'certified secure software lifecycle professional',
        'certified secure software lifecycle professional', 'certified secure software lifecycle professional',
        'certified secure software lifecycle professional', 'certified secure software lifecycle professional'
    }
}

def extract_skills_from_text(text: str, use_nlp: bool = True) -> Dict[str, Set[str]]:
    """
    Extract skills from text using keyword matching and NLP.
    
    Args:
        text: Input text to extract skills from
        use_nlp: Whether to use NLP for enhanced extraction
        
    Returns:
        Dictionary of skill categories with sets of matched skills
    """
    if not text or not isinstance(text, str):
        return {}
        
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    doc = nlp(text_lower) if use_nlp else None
    
    # Initialize result dictionary
    result = {
        'programming': set(),
        'soft_skills': set(),
        'languages': set(),
        'certifications': set(),
        'other': set()
    }
    
    # Extract skills using keyword matching
    for category, subcategories in SKILL_CATEGORIES.items():
        if isinstance(subcategories, dict):
            # Handle nested categories (e.g., programming languages, frameworks)
            for subcategory, skills in subcategories.items():
                for skill in skills:
                    # Check for exact matches and word boundaries
                    pattern = r'\b' + re.escape(skill) + r'\b'
                    if re.search(pattern, text_lower):
                        result['programming'].add(skill)
        else:
            # Handle flat categories (e.g., soft skills, languages)
            for skill in subcategories:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    if category in result:
                        result[category].add(skill)
    
    # Use NLP to extract additional skills
    if use_nlp and doc:
        # Extract noun phrases that might represent skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if len(chunk_text.split()) <= 3:  # Limit to 3-word phrases
                # Check if this phrase is a known skill
                for category, subcategories in SKILL_CATEGORIES.items():
                    if isinstance(subcategories, dict):
                        for skills in subcategories.values():
                            if chunk_text in skills:
                                result['programming'].add(chunk_text)
                                break
                    elif chunk_text in subcategories:
                        if category in result:
                            result[category].add(chunk_text)
    
    # Extract skills with special characters (e.g., C++, C#, .NET)
    special_patterns = [
        r'\b[A-Za-z]+\+\+\b',  # C++
        r'\b[A-Za-z]+#\b',      # C#
        r'\.[A-Za-z]+\b',       # .NET, .js, etc.
        r'\b[A-Za-z]+\s*\d+\b'  # Word followed by numbers (e.g., Python 3, Windows 10)
    ]
    
    for pattern in special_patterns:
        for match in re.finditer(pattern, text):
            skill = match.group(0).strip()
            result['programming'].add(skill)
    
    # Convert sets to lists for JSON serialization
    return {k: list(v) for k, v in result.items() if v}

def extract_job_titles_from_resume(resume_text: str) -> list:
    """
    Extract job titles from resume text using simple pattern matching.
    This is a basic implementation - you might want to enhance it with NLP for better accuracy.
    """
    import re
    
    # Common job title patterns
    patterns = [
        r'(?:^|\n|\b)(?:Senior|Junior|Lead|Staff|Principal)?\s*([A-Z][A-Za-z\s&/]+(?:Engineer|Developer|Programmer|Designer|Analyst|Architect|Manager|Specialist|Consultant|Tester|QA|DevOps|SRE|Data Scientist|ML Engineer|AI Engineer))s?\b',
        r'(?:^|\n|\b)([A-Z][A-Za-z\s&/]+(?:Engineer|Developer|Programmer|Designer|Analyst|Architect|Manager|Specialist|Consultant|Tester|QA|DevOps|SRE|Data Scientist|ML Engineer|AI Engineer))s?\b',
        r'(?:^|\n|\b)(?:Position|Role|Title)[:\s]+([A-Z][A-Za-z\s&/]+(?:Engineer|Developer|Programmer|Designer|Analyst|Architect|Manager|Specialist|Consultant|Tester|QA|DevOps|SRE|Data Scientist|ML Engineer|AI Engineer))s?\b',
    ]
    
    titles = set()
    for pattern in patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            title = match.group(1).strip()
            # Skip very short or generic titles
            if len(title) > 3 and title.lower() not in ['it', 'dev', 'devops', 'qa']:
                titles.add(title)
    
    # Also look for job titles in work experience section
    work_exp_section = re.search(r'(?i)(work\s+experience|experience|employment\s+history)[^\n]*(\n\s*[-*]\s*.*)*', resume_text)
    if work_exp_section:
        exp_text = work_exp_section.group(0)
        # Look for job titles at the start of each line in the experience section
        title_matches = re.finditer(r'(?i)^\s*[-*]?\s*([A-Z][A-Za-z\s&/]+(?:Engineer|Developer|Programmer|Designer|Analyst|Architect|Manager|Specialist|Consultant|Tester|QA|DevOps|SRE|Data Scientist|ML Engineer|AI Engineer))s?', 
                                   exp_text, re.MULTILINE)
        for match in title_matches:
            title = match.group(1).strip()
            if len(title) > 3 and title.lower() not in ['it', 'dev', 'devops', 'qa']:
                titles.add(title)
    
    return list(titles)

def suggest_jobs(resume_text: str, model_name: str = "llama2", limit: int = 10) -> dict:
    """
    Main function to suggest jobs based on resume content.
    
    Args:
        resume_text: Text content of the resume
        model_name: Name of the Ollama model to use for extraction
        limit: Maximum number of jobs to return
        
    Returns:
        dict: Contains extracted resume data and matching jobs
    """
    try:
        logger.info("Starting job suggestion process...")
        
        # Extract skills and experience from resume
        resume_data = extract_skills_and_experience(resume_text, model_name)
        logger.info(f"Extracted resume data: {json.dumps(resume_data, indent=2, ensure_ascii=False)}")
        
        # Extract job titles from resume
        job_titles = resume_data.get("job_titles", []) or extract_job_titles_from_resume(resume_text)
        logger.info(f"Extracted job titles: {job_titles}")
        
        # Extract skills from resume text
        skills = set()
        
        # Get skills from technical_skills if available
        if 'technical_skills' in resume_data and isinstance(resume_data['technical_skills'], list):
            for skill in resume_data['technical_skills']:
                if isinstance(skill, dict) and 'name' in skill:
                    skills.add(skill['name'].lower())
                elif isinstance(skill, str):
                    skills.add(skill.lower())
        
        # Fallback to text extraction if no skills found
        if not skills:
            skills = set(extract_skills_from_text(resume_text))
        
        logger.info(f"Extracted skills: {skills}")
        
        # Get matching jobs from database
        try:
            # Get database connection
            db = get_database()
            if db is None:
                raise Exception("Failed to connect to database: Database object is None")
                
            # Access jobs collection
            jobs_collection = db.get_collection("jobs")
                
            logger.info("Successfully connected to MongoDB and accessed 'jobs' collection")
            
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            return {
                "error": "Failed to connect to database",
                "details": str(e),
                "matching_jobs": [],
                "total_matches": 0,
                "message": "Database connection error"
            }
        
        # Build query based on extracted information
        query = {"status": "active"}  # Only show active jobs
        
        # Add job title filter if we have any
        if job_titles:
            title_regex = "|".join([re.escape(title.lower()) for title in job_titles])
            query["$or"] = [
                {"title": {"$regex": title_regex, "$options": "i"}},
                {"field": {"$regex": title_regex, "$options": "i"}},
                {"technique": {"$regex": title_regex, "$options": "i"}}
            ]
        
        # Add skills filter if we have any
        if skills:
            skills_regex = "|".join([re.escape(skill.lower()) for skill in skills])
            query["$or"] = query.get("$or", []) + [
                {"technique": {"$regex": skills_regex, "$options": "i"}},
                {"description": {"$regex": skills_regex, "$options": "i"}},
                {"type": {"$regex": skills_regex, "$options": "i"}},
                {"field": {"$regex": skills_regex, "$options": "i"}}
            ]
        
        logger.info(f"Querying jobs with: {json.dumps(query, indent=2)}")
        
        # Execute query with projection to only get needed fields
        matching_jobs = list(jobs_collection.find(
            query,
            {
                "_id": 1,
                "title": 1,
                "companyName": 1,
                "location": 1,
                "city": 1,
                "description": 1,
                "technique": 1,
                "field": 1,
                "type": 1,
                "experience": 1,
                "salary": 1,
                "workTime": 1,
                "expiryTime": 1,
                "logoPath": 1,
                "status": 1
            }
        ).limit(limit))
        
        logger.info(f"Found {len(matching_jobs)} potential job matches")
        
        # Score and rank jobs
        scored_jobs = []
        for job in matching_jobs:
            score = 0
            score_breakdown = {}
            
            # Score based on title match (40% weight)
            job_title = (job.get('title') or '').lower()
            job_field = (job.get('field') or '').lower()
            job_technique = (job.get('technique') or '').lower()
            
            if job_titles and (job_title or job_field or job_technique):
                title_score = 0
                for title in job_titles:
                    title = title.lower()
                    # Check against multiple fields
                    for field in [job_title, job_field, job_technique]:
                        if title in field or field in title:
                            title_score = max(title_score, 0.7)  # Partial match
                        if title == field:
                            title_score = 1.0  # Exact match
                            break
                    if title_score == 1.0:
                        break
                
                score += title_score * 40  # 40% weight for title
                score_breakdown['title_match'] = round(title_score * 100, 1)
            
            # Score based on skills match (40% weight)
            job_skills = set()
            # Get skills from technique field (comma-separated)
            if job.get('technique'):
                job_skills.update(s.strip().lower() for s in job['technique'].split(','))
            # Also check description for skills
            if job.get('description'):
                job_skills.update(extract_skills_from_text(job['description']))
            
            if skills and job_skills:
                matched_skills = skills.intersection(job_skills)
                skill_score = len(matched_skills) / max(len(job_skills), 1)
                score += skill_score * 40  # 40% weight for skills
                score_breakdown['skills_match'] = {
                    'matched': list(matched_skills),
                    'score': round(skill_score * 100, 1),
                    'total_required': len(job_skills)
                }
            
            # Add experience level bonus (10%)
            job_exp = (job.get('experience') or '').lower()
            resume_exp = resume_data.get('experience_level', '').lower()
            
            if job_exp and resume_exp:
                # Map experience levels to numerical values for comparison
                exp_levels = {
                    'internship': 0,
                    'entry': 1,
                    'junior': 1,
                    'mid': 2,
                    'senior': 3,
                    'lead': 4,
                    'executive': 5
                }
                
                # Find the closest match for job experience
                job_exp_level = None
                for exp_key in exp_levels:
                    if exp_key in job_exp:
                        job_exp_level = exp_levels[exp_key]
                        break
                
                # Find the closest match for resume experience
                resume_exp_level = exp_levels.get(resume_exp, 2)  # Default to mid-level
                
                if job_exp_level is not None:
                    level_diff = abs(job_exp_level - resume_exp_level)
                    if level_diff == 0:
                        score += 10  # Perfect match
                        score_breakdown['experience_match'] = "Perfect match"
                    elif level_diff == 1:
                        score += 5  # Close match
                        score_breakdown['experience_match'] = "Close match"
                    else:
                        score_breakdown['experience_match'] = f"Experience gap: {level_diff} levels"
            
            # Add bonus for recent jobs (5%)
            if job.get('expiryTime') and job['expiryTime'] > datetime.utcnow():
                days_remaining = (job['expiryTime'] - datetime.utcnow()).days
                if days_remaining <= 7:  # Expiring soon
                    score += 5
                    score_breakdown['urgent_job'] = "Expiring soon"
            
            # Cap score at 100
            score = min(100, round(score, 1))
            
            # Add to results
            scored_jobs.append({
                **job,
                "match_percentage": score,
                "score_breakdown": score_breakdown,
                "matched_skills": list(matched_skills) if 'matched_skills' in locals() else []
            })
        
        # Sort by score (highest first)
        scored_jobs.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Format the response
        result = {
            "matching_jobs": [],
            "total_matches": len(scored_jobs),
            "extracted_data": {
                "job_titles": job_titles,
                "skills": list(skills),
                "experience_level": resume_data.get('experience_level') if isinstance(resume_data, dict) else None
            }
        }
        
        # Add job details to the response
        for job in scored_jobs:
            result["matching_jobs"].append({
                "id": str(job.get("_id")),
                "title": job.get("title", "No Title"),
                "company": job.get("companyName", "Company Not Specified"),
                "location": f"{job.get('city', '')}, {job.get('location', '')}".strip(', '),
                "salary": job.get("salary"),
                "workTime": job.get("workTime"),
                "type": job.get("type"),
                "field": job.get("field"),
                "technique": job.get("technique"),
                "logoPath": job.get("logoPath"),
                "expiryTime": job.get("expiryTime").isoformat() if job.get("expiryTime") else None,
                "match_percentage": job.get("match_percentage", 0),
                "matched_skills": job.get("matched_skills", []),
                "score_breakdown": job.get("score_breakdown", {})
            })
        
        logger.info(f"Job suggestion completed. Found {len(scored_jobs)} matches.")
        return result
        
    except Exception as e:
        error_msg = f"Error in suggest_jobs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": "An error occurred while processing your request",
            "details": str(e),
            "matching_jobs": [],
            "total_matches": 0
        }


# python3.11 -m venv venv
# source venv/bin/activate