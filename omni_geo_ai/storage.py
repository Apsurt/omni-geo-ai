"""Storage module for handling Google Cloud Storage operations."""
import os
from typing import Optional, Tuple
from io import BytesIO
from datetime import datetime

from google.cloud import storage
from PIL import Image

from omni_geo_ai.config import (
    GCS_BUCKET_NAME,
    NORTH_HEMISPHERE_PATH,
    SOUTH_HEMISPHERE_PATH,
    GCS_DRY_RUN,
)


class StorageClient:
    """Client for interacting with Google Cloud Storage."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET_NAME):
        """Initialize the storage client with a bucket name.
        
        Args:
            bucket_name: The name of the GCS bucket to use
        """
        self.bucket_name = bucket_name
        
        if GCS_DRY_RUN:
            print(f"DRY RUN: Storage client initialized with bucket '{bucket_name}'")
            self.client = None
            self.bucket = None
        else:
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
    
    def upload_panorama(
        self, 
        image_data: bytes, 
        latitude: float, 
        longitude: float
    ) -> Tuple[str, str]:
        """Upload a panorama image to Google Cloud Storage.
        
        Args:
            image_data: Raw image data in bytes
            latitude: The latitude where the image was taken
            longitude: The longitude where the image was taken
            
        Returns:
            Tuple containing the GCS path and the image ID
        """
        # Determine hemisphere based on latitude
        hemisphere = "north" if latitude >= 0 else "south"
        hemisphere_path = NORTH_HEMISPHERE_PATH if hemisphere == "north" else SOUTH_HEMISPHERE_PATH
        
        # Generate a unique filename based on coordinates and timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        lat_str = f"{abs(latitude):.6f}{'N' if latitude >= 0 else 'S'}"
        lon_str = f"{abs(longitude):.6f}{'E' if longitude >= 0 else 'W'}"
        image_id = f"{lat_str}_{lon_str}_{timestamp}.jpg"
        
        # Full path in GCS
        gcs_path = f"{hemisphere_path}/{image_id}"
        
        # In dry run mode, we don't actually upload the file
        if GCS_DRY_RUN:
            print(f"DRY RUN: Would upload image to gs://{self.bucket_name}/{gcs_path}")
        else:
            # Upload the image
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(image_data, content_type="image/jpeg")
        
        # Return the GCS path and image ID
        return gcs_path, image_id
    
    def download_panorama(self, gcs_path: str) -> Optional[bytes]:
        """Download a panorama image from Google Cloud Storage.
        
        Args:
            gcs_path: Path to the image in GCS
            
        Returns:
            Image data as bytes, or None if not found
        """
        if GCS_DRY_RUN:
            print(f"DRY RUN: Would download image from gs://{self.bucket_name}/{gcs_path}")
            # Return a small dummy image in dry run mode
            return b"DUMMY_IMAGE_DATA_FOR_DRY_RUN"
        
        blob = self.bucket.blob(gcs_path)
        if not blob.exists():
            return None
        
        return blob.download_as_bytes()
    
    def delete_panorama(self, gcs_path: str) -> bool:
        """Delete a panorama from Google Cloud Storage.
        
        Args:
            gcs_path: Path to the image in GCS
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if GCS_DRY_RUN:
            print(f"DRY RUN: Would delete image at gs://{self.bucket_name}/{gcs_path}")
            return True
            
        blob = self.bucket.blob(gcs_path)
        if not blob.exists():
            return False
        
        blob.delete()
        return True
        
    def clear_all_panoramas(self) -> int:
        """Delete all panorama images from Google Cloud Storage.
        
        Returns:
            Number of deleted images
        """
        if GCS_DRY_RUN:
            print(f"DRY RUN: Would delete all panoramas from gs://{self.bucket_name}")
            return 0
            
        deleted_count = 0
        
        # First delete north hemisphere images
        north_blobs = self.client.list_blobs(
            self.bucket_name, 
            prefix=NORTH_HEMISPHERE_PATH
        )
        for blob in north_blobs:
            blob.delete()
            deleted_count += 1
            
        # Then delete south hemisphere images
        south_blobs = self.client.list_blobs(
            self.bucket_name, 
            prefix=SOUTH_HEMISPHERE_PATH
        )
        for blob in south_blobs:
            blob.delete()
            deleted_count += 1
            
        return deleted_count