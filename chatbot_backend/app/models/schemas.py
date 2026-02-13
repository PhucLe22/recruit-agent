from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CreateUserResp(BaseModel):
    username: str
    created: bool
    message: Optional[str] = None


class UploadResp(BaseModel):
    username: str
    saved: bool
    inserted_id: Optional[str] = None
    message: Optional[str] = None


class UserResp(BaseModel):
    username: str
    email: Optional[str] = None
    created_at: Optional[str] = None
    keywords: Optional[List[str]] = None


class GetUsersResp(BaseModel):
    users: List[UserResp]
    count: int
    message: Optional[str] = None
    parsed_output: Optional[dict] = None


class GoogleMeetRequest(BaseModel):
    summary: str
    description: Optional[str] = None
    start_time: str
    end_time: str
    timezone: str = "UTC"
    attendees: Optional[List[str]] = None


class HealthCheck(BaseModel):
    status: str
    time: datetime
