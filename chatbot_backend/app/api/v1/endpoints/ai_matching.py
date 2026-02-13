import requests
from fastapi import APIRouter, HTTPException, Depends
from app.core.config import settings
from app.core.database import get_database

router = APIRouter()


@router.get("/applicant-matching/{job_id}")
async def get_applicant_matching(job_id: str, db=Depends(get_database)):
    """Get AI-powered applicant matching for a specific job"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.ai_service_api_key
        }
        
        response = requests.get(
            f"{settings.ai_service_base_url}/matching-applicants/{job_id}",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"Service returned status {response.status_code}",
                "applicants": []
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": "AI matching service unavailable",
            "applicants": []
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get applicant matching: {str(e)}")


@router.get("/applicant-matching/all/{business_id}")
async def get_all_applicant_matching(business_id: str, db=Depends(get_database)):
    """Get AI-powered applicant matching for all jobs of a business"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.ai_service_api_key
        }
        
        response = requests.get(
            f"{settings.ai_service_base_url}/matching-applicants/all",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "message": f"Service returned status {response.status_code}",
                "recommendations": {}
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": "AI matching service unavailable",
            "recommendations": {}
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get applicant matching: {str(e)}")


@router.get("/applicant-profile/{user_id}")
async def get_applicant_profile(user_id: str, job_id: str = None, db=Depends(get_database)):
    """Get detailed applicant profile with AI matching insights"""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.ai_service_api_key
        }
        
        url = f"{settings.ai_service_base_url}/applicant/{user_id}/profile"
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
        return {
            "success": False,
            "message": "AI profile service unavailable",
            "data": None
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get applicant profile: {str(e)}")
