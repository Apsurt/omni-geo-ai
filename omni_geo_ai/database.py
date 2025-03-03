"""Database module for the Omni Geo AI project."""
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from omni_geo_ai.config import DATABASE_URL

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Panorama(Base):
    """Represents a panorama image entry in the database."""
    
    __tablename__ = "panoramas"
    
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255), unique=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    hemisphere = Column(String(10), nullable=False, index=True)  # "north" or "south"
    country_code = Column(String(2), nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)


def get_db_session():
    """Create a new database session."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def init_db():
    """Initialize the database by creating all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def close_db(db):
    """Close the database session."""
    db.close()


def clear_db():
    """Clear all data from the database tables without dropping the tables."""
    db = get_db_session()
    try:
        db.query(Panorama).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        close_db(db)