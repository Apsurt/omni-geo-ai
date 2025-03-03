"""Panorama collector for fetching and storing Google Street View images."""
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import hashlib
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from tqdm import tqdm

from omni_geo_ai.config import DAILY_PANORAMA_LIMIT
from omni_geo_ai.database import get_db_session, Panorama, close_db
from omni_geo_ai.storage import StorageClient
from omni_geo_ai.data_collection.streetview_client import StreetViewClient
from omni_geo_ai.data_collection.geocoding import GeocodingService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("panorama_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("panorama_collector")


class PanoramaCollector:
    """Service for collecting and storing panorama images."""
    
    def __init__(self):
        """Initialize the panorama collector service."""
        self.streetview_client = StreetViewClient()
        self.storage_client = StorageClient()
        self.geocoding_service = GeocodingService()
        self.collected_hashes = set()  # For deduplication within a single run
        
    def is_duplicate(self, image_hash: str) -> bool:
        """Check if an image hash is a duplicate.
        
        Args:
            image_hash: MD5 hash of the image
            
        Returns:
            True if duplicate, False otherwise
        """
        # First check in-memory cache for current run
        if image_hash in self.collected_hashes:
            return True
            
        # Then check database
        db = get_db_session()
        try:
            # Check if any panorama in the database has a similar path (contains the hash)
            query = db.query(Panorama).filter(Panorama.image_path.contains(image_hash))
            is_dup = db.query(query.exists()).scalar()
            return is_dup
        finally:
            close_db(db)
        
    def store_panorama(
        self, 
        image_data: bytes, 
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Store a panorama image in GCS and database.
        
        Args:
            image_data: Raw image data in bytes
            metadata: Dictionary with latitude, longitude, etc.
            
        Returns:
            GCS path if successful, None otherwise
        """
        latitude = metadata["latitude"]
        longitude = metadata["longitude"]
        image_hash = metadata["image_hash"]
        
        # Check for duplicates
        if self.is_duplicate(image_hash):
            logger.info(f"Skipping duplicate image at {latitude}, {longitude}")
            return None
            
        # Add to in-memory deduplication cache
        self.collected_hashes.add(image_hash)
        
        # Get country code if not already in metadata
        country_code = metadata.get("country_code")
        if not country_code:
            try:
                country_code = self.geocoding_service.get_country_code(latitude, longitude)
            except Exception as e:
                logger.warning(f"Error getting country code: {e}")
                country_code = None
                
        # Upload to GCS
        try:
            gcs_path, _ = self.storage_client.upload_panorama(
                image_data=image_data,
                latitude=latitude,
                longitude=longitude
            )
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            return None
            
        # Store in database
        db = get_db_session()
        try:
            panorama = Panorama(
                image_path=gcs_path,
                latitude=latitude,
                longitude=longitude,
                hemisphere="north" if latitude >= 0 else "south",
                country_code=country_code,
                date_added=datetime.utcnow()
            )
            
            db.add(panorama)
            db.commit()
            logger.info(f"Stored panorama at {latitude}, {longitude} - Path: {gcs_path}")
            return gcs_path
            
        except IntegrityError:
            db.rollback()
            logger.warning(f"Database integrity error for panorama at {latitude}, {longitude}")
            return None
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing in database: {e}")
            return None
            
        finally:
            close_db(db)
    
    def collect_panoramas(
        self, 
        num_images: int = DAILY_PANORAMA_LIMIT,
        north_ratio: float = 0.5,  # Target ratio of Northern hemisphere images
        max_attempts_per_image: int = 5
    ) -> Dict[str, Any]:
        """Collect a specified number of panorama images.
        
        Args:
            num_images: Number of images to collect
            north_ratio: Target ratio of Northern hemisphere panoramas
            max_attempts_per_image: Max attempts per single successful image
            
        Returns:
            Statistics dictionary with success/failure counts
        """
        logger.info(f"Starting collection of {num_images} panoramas")
        
        # Statistics
        stats = {
            "total_requested": num_images,
            "successfully_stored": 0,
            "failed": 0,
            "duplicates": 0,
            "north_count": 0,
            "south_count": 0,
            "by_country": {},
            "start_time": datetime.utcnow(),
            "end_time": None
        }
        
        # Calculate target counts for each hemisphere
        target_north = int(num_images * north_ratio)
        target_south = num_images - target_north
        
        current_north = 0
        current_south = 0
        
        pbar = tqdm(total=num_images, desc="Collecting panoramas")
        
        while current_north + current_south < num_images:
            # Determine which hemisphere to prioritize
            if current_north >= target_north:
                # We have enough northern, focus on southern
                min_lat, max_lat = -85.0, 0.0
            elif current_south >= target_south:
                # We have enough southern, focus on northern
                min_lat, max_lat = 0.0, 85.0
            else:
                # Still need both, use full range
                min_lat, max_lat = -85.0, 85.0
                
            # Find a valid panorama
            image_data, metadata = self.streetview_client.find_valid_panorama(
                max_attempts=max_attempts_per_image,
                min_lat=min_lat,
                max_lat=max_lat
            )
            
            if image_data is None or metadata is None:
                stats["failed"] += 1
                logger.warning("Failed to find a valid panorama")
                continue
                
            # Check for duplicates before storage
            if self.is_duplicate(metadata["image_hash"]):
                stats["duplicates"] += 1
                logger.info("Skipping duplicate panorama")
                continue
                
            # Store the panorama
            gcs_path = self.store_panorama(image_data, metadata)
            
            if gcs_path:
                # Successfully stored
                is_north = metadata["latitude"] >= 0
                hemisphere = "north" if is_north else "south"
                
                if is_north:
                    current_north += 1
                    stats["north_count"] += 1
                else:
                    current_south += 1
                    stats["south_count"] += 1
                    
                stats["successfully_stored"] += 1
                
                # Update country statistics if available
                country_code = metadata.get("country_code")
                if country_code:
                    if country_code not in stats["by_country"]:
                        stats["by_country"][country_code] = 0
                    stats["by_country"][country_code] += 1
                    
                pbar.update(1)
                logger.info(f"Stored {hemisphere} panorama ({stats['successfully_stored']}/{num_images})")
                
                # Throttle to avoid API rate limits
                time.sleep(0.5)
            else:
                stats["failed"] += 1
                logger.warning("Failed to store panorama")
                
        pbar.close()
        stats["end_time"] = datetime.utcnow()
        elapsed = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        logger.info(f"Collection complete. Collected {stats['successfully_stored']} panoramas in {elapsed:.1f} seconds")
        logger.info(f"North: {stats['north_count']}, South: {stats['south_count']}")
        
        return stats