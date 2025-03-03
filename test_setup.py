#!/usr/bin/env python3
"""
Simple test script to verify that our environment is set up correctly.
"""
import os
import sys

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import omni_geo_ai.config as config
from omni_geo_ai.database import init_db, get_db_session, close_db, Panorama

def test_environment():
    """Test that the environment is set up correctly."""
    
    print(f"Using SQLite: {config.USE_SQLITE}")
    print(f"Database URL: {config.DATABASE_URL}")
    
    # Test database connection
    print("\nInitializing database...")
    init_db()
    
    # Test session creation
    db = get_db_session()
    print("Database session created successfully")
    
    # Check that tables were created
    panorama_count = db.query(Panorama).count()
    print(f"Number of panoramas in database: {panorama_count}")
    
    # Clean up
    close_db(db)
    print("Database session closed")
    
    print("\nEnvironment is set up correctly!")

if __name__ == "__main__":
    test_environment()