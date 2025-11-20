"""Authentication models."""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from n3_server.db.session import Base


class Role(str, enum.Enum):
    """User roles."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(String(50), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    owned_projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    project_members = relationship("ProjectMember", back_populates="user", cascade="all, delete-orphan")
    created_shares = relationship("ShareLink", foreign_keys="ShareLink.created_by_user_id")


class ProjectMember(Base):
    """Project membership and permissions."""
    __tablename__ = "project_members"
    
    id = Column(String(50), primary_key=True)
    project_id = Column(String(50), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(Role), nullable=False, default=Role.VIEWER)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_members")
    
    __table_args__ = (
        # Ensure unique user per project
        {'sqlite_autoincrement': True}
    )
