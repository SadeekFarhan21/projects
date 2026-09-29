import httpx
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.models.database import get_db, User
from datetime import datetime
from typing import Optional

security = HTTPBearer()

class AuthService:
    """Service for handling Google OAuth authentication"""
    
    @staticmethod
    async def verify_google_token(token: str) -> dict:
        """Verify Google OAuth token and return user info"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token}"
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid token")
                return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    @staticmethod
    async def get_or_create_user(user_info: dict, db: Session) -> User:
        """Get existing user or create new one from Google OAuth info"""
        user_id = user_info["id"]
        
        # Try to get existing user
        user = db.query(User).filter(User.id == user_id).first()
        
        if user:
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        
        # Create new user
        user = User(
            id=user_id,
            name=user_info["name"],
            email=user_info["email"],
            picture=user_info.get("picture", ""),
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    
    # Verify token with Google
    user_info = await AuthService.verify_google_token(token)
    
    # Get or create user in database
    user = await AuthService.get_or_create_user(user_info, db)
    
    return user

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Optional dependency to get current user (for endpoints that work with or without auth)"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None
