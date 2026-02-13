from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.core.database import get_database
from app.models.schemas import UserResp, GetUsersResp, CreateUserResp, UploadResp

router = APIRouter()


@router.get("/", response_model=GetUsersResp)
async def get_users(db=Depends(get_database)):
    """Get all users"""
    try:
        users = await db.users.find().to_list(length=None)
        user_list = [
            UserResp(
                username=user.get("username", ""),
                email=user.get("email"),
                created_at=user.get("created_at"),
                keywords=user.get("keywords", [])
            )
            for user in users
        ]
        
        return GetUsersResp(
            users=user_list,
            count=len(user_list),
            message="Users retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=CreateUserResp)
async def create_user(user_data: dict, db=Depends(get_database)):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = await db.users.find_one({"username": user_data.get("username")})
        if existing_user:
            return CreateUserResp(
                username=user_data.get("username"),
                created=False,
                message="User already exists"
            )
        
        # Create new user
        user_data["created_at"] = datetime.utcnow().isoformat()
        result = await db.users.insert_one(user_data)
        
        return CreateUserResp(
            username=user_data.get("username"),
            created=True,
            message="User created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{username}", response_model=UserResp)
async def get_user(username: str, db=Depends(get_database)):
    """Get user by username"""
    try:
        user = await db.users.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResp(
            username=user.get("username", ""),
            email=user.get("email"),
            created_at=user.get("created_at"),
            keywords=user.get("keywords", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
