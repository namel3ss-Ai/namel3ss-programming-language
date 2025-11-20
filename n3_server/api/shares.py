from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from jose import jwt
from nanoid import generate

from n3_server.db.session import get_db
from n3_server.db.models import Project, ShareLink
from n3_server.config import get_settings

router = APIRouter()
settings = get_settings()


class CreateShareRequest(BaseModel):
    role: str  # viewer, editor
    expiresInHours: int | None = None


class ShareLinkResponse(BaseModel):
    id: str
    projectId: str
    token: str
    url: str
    role: str
    createdAt: datetime
    expiresAt: datetime | None
    createdByUserId: str | None = None


class ShareTokenValidation(BaseModel):
    projectId: str
    role: str


@router.post("/{project_id}/shares", response_model=ShareLinkResponse)
async def create_share(
    project_id: str,
    request: CreateShareRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a share link for a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if request.role not in ["viewer", "editor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    share_id = generate(size=12)
    token = generate(size=32)
    expires_at = None
    
    if request.expiresInHours:
        expires_at = datetime.utcnow() + timedelta(hours=request.expiresInHours)
    
    share_link = ShareLink(
        id=share_id,
        project_id=project_id,
        token=token,
        role=request.role,
        expires_at=expires_at,
    )
    
    db.add(share_link)
    await db.commit()
    await db.refresh(share_link)
    
    return ShareLinkResponse(
        id=share_link.id,
        projectId=share_link.project_id,
        token=share_link.token,
        url=f"/open/{token}",
        role=share_link.role,
        createdAt=share_link.created_at,
        expiresAt=share_link.expires_at,
    )


@router.get("/{project_id}/shares", response_model=list[ShareLinkResponse])
async def list_shares(project_id: str, db: AsyncSession = Depends(get_db)):
    """List all share links for a project."""
    result = await db.execute(
        select(ShareLink).where(ShareLink.project_id == project_id)
    )
    shares = result.scalars().all()
    
    return [
        ShareLinkResponse(
            id=s.id,
            projectId=s.project_id,
            token=s.token,
            url=f"/open/{s.token}",
            role=s.role,
            createdAt=s.created_at,
            expiresAt=s.expires_at,
        )
        for s in shares
    ]


@router.delete("/{project_id}/shares/{share_id}")
async def delete_share(
    project_id: str,
    share_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a share link."""
    result = await db.execute(
        select(ShareLink).where(
            ShareLink.id == share_id,
            ShareLink.project_id == project_id,
        )
    )
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Share not found")
    
    await db.delete(share)
    await db.commit()
    
    return {"status": "deleted"}


@router.get("/open-by-token", response_model=ShareTokenValidation)
async def validate_token(token: str = Query(...), db: AsyncSession = Depends(get_db)):
    """Validate a share token and return project access."""
    result = await db.execute(select(ShareLink).where(ShareLink.token == token))
    share = result.scalar_one_or_none()
    
    if not share:
        raise HTTPException(status_code=404, detail="Invalid token")
    
    # Check expiration
    if share.expires_at and share.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Token expired")
    
    return ShareTokenValidation(
        projectId=share.project_id,
        role=share.role,
    )
