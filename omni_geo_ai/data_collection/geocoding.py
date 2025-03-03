"""Geocoding utilities for converting coordinates to country codes."""
from typing import Dict, Optional, Any
import time

import requests

from omni_geo_ai.config import GOOGLE_API_KEY


class GeocodingService:
    """Service for reverse geocoding coordinates to country information."""
    
    def __init__(self, api_key: str = GOOGLE_API_KEY):
        """Initialize with Google API key.
        
        Args:
            api_key: Google Maps API key with Geocoding API enabled
        """
        self.api_key = api_key
        self.geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        
        # Cache to minimize API calls
        self.cache = {}
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Convert latitude and longitude to address information.
        
        Args:
            latitude: The latitude coordinate
            longitude: The longitude coordinate
            
        Returns:
            Dictionary containing address components or error info
        """
        # Check cache first
        cache_key = f"{latitude:.6f},{longitude:.6f}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        params = {
            "latlng": f"{latitude},{longitude}",
            "key": self.api_key
        }
        
        response = requests.get(self.geocode_url, params=params)
        if response.status_code != 200:
            return {"status": "ERROR", "error": f"API error {response.status_code}"}
            
        data = response.json()
        if data["status"] != "OK":
            return {"status": data["status"]}
            
        # Store in cache
        self.cache[cache_key] = data
        
        return data
    
    def get_country_code(self, latitude: float, longitude: float) -> Optional[str]:
        """Extract the country code from geocoding results.
        
        Args:
            latitude: The latitude coordinate
            longitude: The longitude coordinate
            
        Returns:
            Two-letter country code (ISO 3166-1 alpha-2) or None if not found
        """
        geocode_data = self.reverse_geocode(latitude, longitude)
        
        if geocode_data.get("status") != "OK":
            return None
            
        results = geocode_data.get("results", [])
        if not results:
            return None
            
        # Look for country in address components
        for result in results:
            for component in result.get("address_components", []):
                if "country" in component.get("types", []):
                    return component.get("short_name")
                    
        return None
    
    def get_address_components(
        self, 
        latitude: float, 
        longitude: float
    ) -> Dict[str, str]:
        """Get a dictionary of address components from coordinates.
        
        Args:
            latitude: The latitude coordinate
            longitude: The longitude coordinate
            
        Returns:
            Dictionary with country_code, locality, and other components
        """
        geocode_data = self.reverse_geocode(latitude, longitude)
        components = {}
        
        if geocode_data.get("status") != "OK":
            return components
            
        results = geocode_data.get("results", [])
        if not results:
            return components
            
        # Extract components from the most detailed result
        result = results[0]
        
        for component in result.get("address_components", []):
            types = component.get("types", [])
            
            if "country" in types:
                components["country_code"] = component.get("short_name")
                components["country"] = component.get("long_name")
                
            elif "locality" in types:
                components["locality"] = component.get("long_name")
                
            elif "administrative_area_level_1" in types:
                components["admin_area"] = component.get("long_name")
                
        return components