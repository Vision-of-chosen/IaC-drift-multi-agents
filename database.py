#!/usr/bin/env python3
"""
Database models and setup for the multi-user Terraform Drift Detection & Remediation System.

This module provides database models for users, conversations, and messages,
enabling persistent storage of user interactions across sessions.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///terraform_drift_system.db")

class User(Base):
    """User model - uses AWS_ACCESS_KEY_ID as primary identifier"""
    __tablename__ = "users"
    
    aws_access_key_id = Column(String(20), primary_key=True, index=True)
    aws_region = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(aws_access_key_id='{self.aws_access_key_id[:8]}...', aws_region='{self.aws_region}')>"


class Conversation(Base):
    """Conversation model - represents a chat session between user and system"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20), ForeignKey("users.aws_access_key_id"), nullable=False)
    title = Column(String(255), nullable=True)  # Optional conversation title
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default="active")  # active, completed, archived
    conversation_state = Column(String(50), default="idle")  # idle, detection_complete, analysis_complete, etc.
    context = Column(JSON, default=dict)  # Store conversation context as JSON
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id[:8]}...', status='{self.status}')>"


class Message(Base):
    """Message model - represents individual messages in a conversation"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    routed_agent = Column(String(50), nullable=True)  # Which agent was used, if any
    agent_result = Column(Text, nullable=True)  # Result from agent execution
    created_at = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, default=dict)  # Additional message metadata
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"


class DatabaseManager:
    """Database manager for handling all database operations"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def create_or_get_user(self, aws_access_key_id: str, aws_region: str) -> User:
        """Create a new user or get existing user"""
        with self.get_session() as session:
            try:
                # Try to get existing user
                user = session.query(User).filter(User.aws_access_key_id == aws_access_key_id).first()
                
                if user:
                    # Update last active time and region if changed
                    user.last_active = datetime.utcnow()
                    user.aws_region = aws_region
                    user.is_active = True
                else:
                    # Create new user
                    user = User(
                        aws_access_key_id=aws_access_key_id,
                        aws_region=aws_region
                    )
                    session.add(user)
                
                session.commit()
                session.refresh(user)
                logger.info(f"User created/updated: {aws_access_key_id[:8]}...")
                return user
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error creating/getting user: {e}")
                raise
    
    def create_conversation(self, user_id: str, title: str = None) -> Conversation:
        """Create a new conversation for a user"""
        with self.get_session() as session:
            try:
                conversation = Conversation(
                    user_id=user_id,
                    title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                )
                session.add(conversation)
                session.commit()
                session.refresh(conversation)
                logger.info(f"Conversation created: {conversation.id} for user {user_id[:8]}...")
                return conversation
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error creating conversation: {e}")
                raise
    
    def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Conversation]:
        """Get recent conversations for a user"""
        with self.get_session() as session:
            try:
                conversations = session.query(Conversation)\
                    .filter(Conversation.user_id == user_id)\
                    .order_by(Conversation.updated_at.desc())\
                    .limit(limit)\
                    .all()
                return conversations
                
            except SQLAlchemyError as e:
                logger.error(f"Error getting user conversations: {e}")
                return []
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a conversation by ID"""
        with self.get_session() as session:
            try:
                return session.query(Conversation).filter(Conversation.id == conversation_id).first()
            except SQLAlchemyError as e:
                logger.error(f"Error getting conversation: {e}")
                return None
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   routed_agent: str = None, agent_result: str = None, 
                   message_metadata: Dict = None) -> Message:
        """Add a message to a conversation"""
        with self.get_session() as session:
            try:
                message = Message(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    routed_agent=routed_agent,
                    agent_result=agent_result,
                    message_metadata=message_metadata or {}
                )
                session.add(message)
                
                # Update conversation updated_at timestamp
                conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
                if conversation:
                    conversation.updated_at = datetime.utcnow()
                
                session.commit()
                session.refresh(message)
                return message
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error adding message: {e}")
                raise
    
    def update_conversation_state(self, conversation_id: int, state: str, context: Dict = None):
        """Update conversation state and context"""
        with self.get_session() as session:
            try:
                conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
                if conversation:
                    conversation.conversation_state = state
                    conversation.updated_at = datetime.utcnow()
                    if context:
                        conversation.context = context
                    session.commit()
                    logger.info(f"Conversation {conversation_id} state updated to: {state}")
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating conversation state: {e}")
                raise
    
    def get_conversation_messages(self, conversation_id: int, limit: int = 50) -> List[Message]:
        """Get messages for a conversation"""
        with self.get_session() as session:
            try:
                messages = session.query(Message)\
                    .filter(Message.conversation_id == conversation_id)\
                    .order_by(Message.created_at)\
                    .limit(limit)\
                    .all()
                return messages
                
            except SQLAlchemyError as e:
                logger.error(f"Error getting conversation messages: {e}")
                return []
    
    def get_user_by_aws_key(self, aws_access_key_id: str) -> Optional[User]:
        """Get user by AWS access key ID"""
        with self.get_session() as session:
            try:
                return session.query(User).filter(User.aws_access_key_id == aws_access_key_id).first()
            except SQLAlchemyError as e:
                logger.error(f"Error getting user by AWS key: {e}")
                return None
    
    def cleanup_old_conversations(self, days: int = 30):
        """Archive old conversations (optional maintenance function)"""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                session.query(Conversation)\
                    .filter(Conversation.updated_at < cutoff_date)\
                    .update({"status": "archived"})
                session.commit()
                logger.info(f"Archived conversations older than {days} days")
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error archiving old conversations: {e}")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session():
    """Helper function to get database session - useful for dependency injection"""
    return db_manager.get_session()


def init_database():
    """Initialize database and create tables"""
    try:
        db_manager._create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise 