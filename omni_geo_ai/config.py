"""Configuration settings for the Omni Geo AI project."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MAPS_STREETVIEW_API_KEY = os.getenv("GOOGLE_MAPS_STREETVIEW_API_KEY") or GOOGLE_API_KEY
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

# Google Cloud Storage
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_DRY_RUN = os.getenv("GCS_DRY_RUN", "false").lower() == "true"

# PostgreSQL Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "omni_geo_ai")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Data Collection Settings
DAILY_PANORAMA_LIMIT = int(os.getenv("DAILY_PANORAMA_LIMIT", "1000"))

# Database URL for SQLAlchemy
# Check if we should use SQLite for local development
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

if USE_SQLITE:
    DATABASE_URL = "sqlite:///omni_geo_ai.db"
else:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Street View API parameters
STREETVIEW_IMAGE_SIZE = "640x640"
STREETVIEW_API_URL = "https://maps.googleapis.com/maps/api/streetview"
STREETVIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# GCS paths for storing panoramas
NORTH_HEMISPHERE_PATH = "hemisphere/north"
SOUTH_HEMISPHERE_PATH = "hemisphere/south"