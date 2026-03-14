# app/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    
    # Secret key for session encryption
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///cricket.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True to see SQL queries (useful for debugging)
    
    # Redis configuration (for caching - we'll use this later)
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # SocketIO configuration (for real-time updates)
    SOCKETIO_ASYNC_MODE = 'eventlet'
    
    # Pagination
    ITEMS_PER_PAGE = 20
    
    # API rate limiting (requests per minute)
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100/minute"

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Show SQL queries in console
    CACHE_TYPE = 'SimpleCache'  # No Redis needed locally

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    # Override with production database URL
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # In-memory database for tests
    CACHE_TYPE = 'simple'  # Don't need Redis for tests

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}