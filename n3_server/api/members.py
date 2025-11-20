"""Project membership and permissions API."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from nanoid import generate

from n3_server.db.session import get_db
from n3_server.auth import (
    User,
    ProjectMember,
    Role,
    get_current_active_user,
    require_owner_access,
)
from n3_server.db.models import Project

router = APIRouter(prefix="/projects/{project_id}/members", tags=["project-members"])


# Request/Response Models
class MemberAdd(BaseModel):
    """Add member request."""
    email: EmailStr
    role: Role = Role.VIEWER


class MemberUpdate(BaseModel):
    """Update member request."""
    role: Role


class MemberResponse(BaseModel):
    """Member response."""
    id: str
    user_id: str
    user_email: str
    user_username: str
    role: Role
    created_at: str
    
    class Config:
        from_attributes = True


# Endpoints
@router.get("", response_model=List[MemberResponse])
async def list_members(
    project_id: str,
    current_user: User = Depends(require_owner_access()),
    db: AsyncSession = Depends(get_db),
):
    """
    List all members of a project.
    
    Requires owner access.
    """
    result = await db.execute(
        select(ProjectMember, User).join(
            User, ProjectMember.user_id == User.id
        ).where(
            ProjectMember.project_id == project_id
        )
    )
    
    members_data = []
    for membership, user in result.all():
        members_data.append({
            "id": membership.id,
            "user_id": user.id,
            "user_email": user.email,
            "user_username": user.username,
            "role": membership.role,
            "created_at": membership.created_at.isoformat(),
        })
    
    return members_data


@router.post("", response_model=MemberResponse, status_code=status.HTTP_201_CREATED)
async def add_member(
    project_id: str,
    member_data: MemberAdd,
    current_user: User = Depends(require_owner_access()),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a member to a project.
    
    Requires owner access.
    """
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == member_data.email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found with that email",
        )
    
    # Check if user is already a member
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user.id,
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of this project",
        )
    
    # Create membership
    membership = ProjectMember(
        id=generate(size=12),
        project_id=project_id,
        user_id=user.id,
        role=member_data.role,
    )
    
    db.add(membership)
    await db.commit()
    await db.refresh(membership)
    
    return {
        "id": membership.id,
        "user_id": user.id,
        "user_email": user.email,
        "user_username": user.username,
        "role": membership.role,
        "created_at": membership.created_at.isoformat(),
    }


@router.patch("/{member_id}", response_model=MemberResponse)
async def update_member(
    project_id: str,
    member_id: str,
    member_update: MemberUpdate,
    current_user: User = Depends(require_owner_access()),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a member's role.
    
    Requires owner access.
    """
    # Get membership
    result = await db.execute(
        select(ProjectMember, User).join(
            User, ProjectMember.user_id == User.id
        ).where(
            ProjectMember.id == member_id,
            ProjectMember.project_id == project_id,
        )
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )
    
    membership, user = row
    
    # Update role
    membership.role = member_update.role
    await db.commit()
    await db.refresh(membership)
    
    return {
        "id": membership.id,
        "user_id": user.id,
        "user_email": user.email,
        "user_username": user.username,
        "role": membership.role,
        "created_at": membership.created_at.isoformat(),
    }


@router.delete("/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    project_id: str,
    member_id: str,
    current_user: User = Depends(require_owner_access()),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a member from a project.
    
    Requires owner access.
    """
    # Get membership
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.id == member_id,
            ProjectMember.project_id == project_id,
        )
    )
    membership = result.scalar_one_or_none()
    
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )
    
    # Delete membership
    await db.delete(membership)
    await db.commit()
    
    return None
