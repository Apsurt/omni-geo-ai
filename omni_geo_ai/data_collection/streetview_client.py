"""Client for Google Street View API to fetch panoramas."""
import random
import time
from typing import Dict, Tuple, Optional, List, Any
import hashlib

import requests

from omni_geo_ai.config import (
    GOOGLE_MAPS_STREETVIEW_API_KEY,
    STREETVIEW_API_URL,
    STREETVIEW_METADATA_URL,
    STREETVIEW_IMAGE_SIZE,
)


class StreetViewClient:
    """Client for interacting with Google Street View API."""
    
    def __init__(self, api_key: str = ""):
        """Initialize with Google API key.
        
        Args:
            api_key: Google Maps API key with Street View Static API enabled
        """
        api_key_value = api_key or GOOGLE_MAPS_STREETVIEW_API_KEY
        if not api_key_value:
            raise ValueError("API key is required but not provided and not found in environment")
        self.api_key: str = api_key_value
    
    def get_random_coordinates(
        self, 
        min_lat: float = -85.0, 
        max_lat: float = 85.0,
        min_lon: float = -180.0, 
        max_lon: float = 180.0
    ) -> Tuple[float, float]:
        """Generate random coordinates within the given bounds.
        
        Args:
            min_lat: Minimum latitude (-85.0 by default)
            max_lat: Maximum latitude (85.0 by default)
            min_lon: Minimum longitude (-180.0 by default)
            max_lon: Maximum longitude (180.0 by default)
            
        Returns:
            Tuple of (latitude, longitude)
        """
        latitude = random.uniform(min_lat, max_lat)
        longitude = random.uniform(min_lon, max_lon)
        return latitude, longitude
    
    def check_panorama_exists(self, latitude: float, longitude: float) -> bool:
        """Check if a Street View panorama exists at the given coordinates.
        
        Args:
            latitude: Latitude to check
            longitude: Longitude to check
            
        Returns:
            True if a panorama exists, False otherwise
        """
        params: dict[str, str] = {
            "location": f"{latitude},{longitude}",
            "key": self.api_key
        }
        
        response = requests.get(STREETVIEW_METADATA_URL, params=params)
        if response.status_code != 200:
            return False
        
        data = response.json()
        return data.get("status") == "OK"
    
    def fetch_panorama(
        self, 
        latitude: float, 
        longitude: float,
        heading: Optional[float] = None,
        fov: float = 90.0
    ) -> Optional[bytes]:
        """Fetch a panorama image from Street View.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            heading: Optional camera heading (0-360 degrees, random if None)
            fov: Field of view (default 90 degrees)
            
        Returns:
            Image data as bytes if successful, None otherwise
        """
        # Use a random heading if not specified
        if heading is None:
            heading = random.uniform(0, 360)
            
        params: dict[str, str | float] = {
            "location": f"{latitude},{longitude}",
            "size": STREETVIEW_IMAGE_SIZE,
            "heading": heading,
            "fov": fov,
            "key": self.api_key
        }
        
        response = requests.get(STREETVIEW_API_URL, params=params)
        if response.status_code != 200:
            return None
            
        # Check if the response is an actual image (not the "no image" placeholder)
        content_type = response.headers.get("content-type", "")
        if "image/jpeg" not in content_type:
            return None
            
        return response.content
    
    def fetch_panorama_with_metadata(
        self, 
        latitude: float, 
        longitude: float,
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """Fetch a panorama image and its metadata.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Tuple containing image data (or None) and metadata dict
        """
        # First check if panorama exists
        if not self.check_panorama_exists(latitude, longitude):
            return None, {"status": "NOT_FOUND"}
        
        # Then fetch the actual image
        image_data = self.fetch_panorama(latitude, longitude)
        if image_data is None:
            return None, {"status": "ERROR"}
        
        # Generate image hash to help with deduplication
        image_hash = hashlib.md5(image_data).hexdigest()
        
        metadata = {
            "status": "OK",
            "latitude": latitude,
            "longitude": longitude,
            "hemisphere": "north" if latitude >= 0 else "south",
            "image_hash": image_hash,
        }
        
        return image_data, metadata
    
    def find_valid_panorama(
        self,
        max_attempts: int = 10,
        min_lat: float = -85.0, 
        max_lat: float = 85.0,
        min_lon: float = -180.0, 
        max_lon: float = 180.0
    ) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """Find a valid panorama by trying random coordinates.
        
        Args:
            max_attempts: Maximum number of attempts
            min_lat: Minimum latitude (-85.0 by default)
            max_lat: Maximum latitude (85.0 by default)
            min_lon: Minimum longitude (-180.0 by default)
            max_lon: Maximum longitude (180.0 by default)
            
        Returns:
            Tuple of (image_data, metadata) if found, (None, None) otherwise
        """
        for _ in range(max_attempts):
            lat, lon = self.get_random_coordinates(min_lat, max_lat, min_lon, max_lon)
            image_data, metadata = self.fetch_panorama_with_metadata(lat, lon)
            
            if image_data is not None and metadata["status"] == "OK":
                return image_data, metadata
                
            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.2)
            
        return None, None