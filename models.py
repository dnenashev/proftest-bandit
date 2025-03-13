from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime
import os
import json

Base = declarative_base()

class Question(Base):
    """Model for storing test questions"""
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    text = Column(String(500), nullable=False)
    answers = Column(JSON, nullable=False)  # Store answers as JSON array
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    positions = relationship("QuestionPosition", back_populates="question")
    interactions = relationship("QuestionInteraction", back_populates="question")
    
    def __repr__(self):
        return f"<Question(id={self.id}, text='{self.text[:30]}...')>"

class QuestionPosition(Base):
    """Model for tracking question positions in tests"""
    __tablename__ = 'question_positions'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey('questions.id'))
    position = Column(Integer, nullable=False)  # Position in the test (0-based index)
    impressions = Column(Integer, default=0)  # Number of times shown at this position
    conversions = Column(Integer, default=0)  # Number of conversions when shown at this position
    reward = Column(Float, default=0.0)  # Current reward estimate (conversion rate)
    
    # Relationships
    question = relationship("Question", back_populates="positions")
    
    def __repr__(self):
        return f"<QuestionPosition(question_id={self.question_id}, position={self.position}, reward={self.reward})>"

class TestSession(Base):
    """Model for tracking user test sessions"""
    __tablename__ = 'test_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    question_order = Column(JSON, nullable=False)  # Store question IDs as JSON array
    completed = Column(Boolean, default=False)
    converted = Column(Boolean, default=False)  # Whether user submitted the form
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    interactions = relationship("QuestionInteraction", back_populates="session")
    
    def __repr__(self):
        return f"<TestSession(id={self.id}, completed={self.completed}, converted={self.converted})>"

class QuestionInteraction(Base):
    """Model for tracking user interactions with questions"""
    __tablename__ = 'question_interactions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('test_sessions.id'))
    question_id = Column(Integer, ForeignKey('questions.id'))
    position = Column(Integer, nullable=False)  # Position where question was shown
    answer_index = Column(Integer, nullable=True)  # Which answer was selected (0-based index)
    time_spent = Column(Integer, nullable=True)  # Time spent on question in seconds
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    session = relationship("TestSession", back_populates="interactions")
    question = relationship("Question", back_populates="interactions")
    
    def __repr__(self):
        return f"<QuestionInteraction(session_id={self.session_id}, question_id={self.question_id}, position={self.position})>"

# Database initialization function
def init_db(db_path=None):
    # Use DATABASE_URL from environment variables if available
    from dotenv import load_dotenv
    load_dotenv()
    
    if db_path is None:
        db_path = os.environ.get('DATABASE_URL', 'sqlite:///bandit_test.db')
        
    # Handle PostgreSQL connection string for Heroku
    if db_path.startswith('postgres://'):
        db_path = db_path.replace('postgres://', 'postgresql://', 1)
        
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()