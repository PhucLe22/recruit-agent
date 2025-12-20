from pydantic import BaseModel
from typing import Optional

MODEL_NAME = "qwen2.5:3b"

# ---------- Pydantic responses ----------
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
    keywords: Optional[list] = None

class GetUsersResp(BaseModel):
    users: list[UserResp]
    count: int
    message: Optional[str] = None
    parsed_output: Optional[dict] = None
