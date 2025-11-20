from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Text, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from n3_server.db.session import Base


class Project(Base):
    """Graph project storage."""
    __tablename__ = "projects"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    owner_id = Column(String(50), ForeignKey("users.id"), nullable=True)  # Nullable for backward compatibility
    graph_data = Column(JSON, nullable=False, default={})
    project_metadata = Column(JSON, default={})  # Renamed from 'metadata' (reserved by SQLAlchemy)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="owned_projects")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    shares = relationship("ShareLink", back_populates="project", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="project", cascade="all, delete-orphan")


class ShareLink(Base):
    """Share links for collaboration."""
    __tablename__ = "share_links"
    
    id = Column(String(50), primary_key=True)
    project_id = Column(String(50), ForeignKey("projects.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # viewer, editor
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    created_by_user_id = Column(String(50), nullable=True)
    
    project = relationship("Project", back_populates="shares")


class Feedback(Base):
    """Feedback for RLHF training."""
    __tablename__ = "feedback"
    
    id = Column(String(50), primary_key=True)
    project_id = Column(String(50), ForeignKey("projects.id"), nullable=False)
    agent_id = Column(String(50), nullable=False, index=True)
    run_id = Column(String(50), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="feedbacks")


class PolicyVersion(Base):
    """Trained policy versions."""
    __tablename__ = "policy_versions"
    
    id = Column(String(50), primary_key=True)
    agent_id = Column(String(50), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_path = Column(String(500), nullable=False)
    feedback_count = Column(Integer, default=0)
    reward_mean = Column(Float, nullable=True)
    reward_std = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
