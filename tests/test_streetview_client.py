"""Tests for the StreetViewClient class."""
import unittest
from unittest.mock import patch, MagicMock
import hashlib

from omni_geo_ai.data_collection.streetview_client import StreetViewClient


class TestStreetViewClient(unittest.TestCase):
    """Test cases for the StreetViewClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = StreetViewClient(api_key="test_api_key")
    
    def test_get_random_coordinates(self):
        """Test that random coordinates are within specified bounds."""
        for _ in range(10):  # Test multiple times for randomness
            lat, lon = self.client.get_random_coordinates(
                min_lat=-10.0, max_lat=10.0,
                min_lon=-20.0, max_lon=20.0
            )
            
            self.assertGreaterEqual(lat, -10.0)
            self.assertLessEqual(lat, 10.0)
            self.assertGreaterEqual(lon, -20.0)
            self.assertLessEqual(lon, 20.0)
    
    @patch('requests.get')
    def test_check_panorama_exists_true(self, mock_get):
        """Test check_panorama_exists when panorama exists."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.check_panorama_exists(37.7749, -122.4194)
        
        # Verify
        self.assertTrue(result)
        mock_get.assert_called_once()
        
    @patch('requests.get')
    def test_check_panorama_exists_false(self, mock_get):
        """Test check_panorama_exists when panorama doesn't exist."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.check_panorama_exists(0, 0)
        
        # Verify
        self.assertFalse(result)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_fetch_panorama_success(self, mock_get):
        """Test fetch_panorama when successful."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"test_image_data"
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.fetch_panorama(37.7749, -122.4194, heading=90.0)
        
        # Verify
        self.assertEqual(result, b"test_image_data")
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_fetch_panorama_failure(self, mock_get):
        """Test fetch_panorama when API fails."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.fetch_panorama(37.7749, -122.4194)
        
        # Verify
        self.assertIsNone(result)
        mock_get.assert_called_once()
    
    @patch.object(StreetViewClient, 'check_panorama_exists')
    @patch.object(StreetViewClient, 'fetch_panorama')
    def test_fetch_panorama_with_metadata(self, mock_fetch, mock_check):
        """Test fetch_panorama_with_metadata when successful."""
        # Mock the dependency methods
        mock_check.return_value = True
        mock_fetch.return_value = b"test_image_data"
        
        # Test
        image_data, metadata = self.client.fetch_panorama_with_metadata(40.7128, -74.0060)
        
        # Verify
        self.assertEqual(image_data, b"test_image_data")
        self.assertEqual(metadata["status"], "OK")
        self.assertEqual(metadata["latitude"], 40.7128)
        self.assertEqual(metadata["longitude"], -74.0060)
        self.assertEqual(metadata["hemisphere"], "north")
        self.assertEqual(metadata["image_hash"], hashlib.md5(b"test_image_data").hexdigest())
        
    @patch.object(StreetViewClient, 'check_panorama_exists')
    def test_fetch_panorama_with_metadata_not_found(self, mock_check):
        """Test fetch_panorama_with_metadata when panorama doesn't exist."""
        # Mock the dependency method
        mock_check.return_value = False
        
        # Test
        image_data, metadata = self.client.fetch_panorama_with_metadata(0, 0)
        
        # Verify
        self.assertIsNone(image_data)
        self.assertEqual(metadata["status"], "NOT_FOUND")


if __name__ == '__main__':
    unittest.main()