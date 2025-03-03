"""Client for Google Street View API to fetch panoramas."""
import random
import time
import math
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
    
    # Cache of known good spots (will populate dynamically)
    _successful_locations = []
    _max_cache_size = 1000
    
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
        # Known cities with very high Street View coverage
        # This list can be auto-updated by running: python scripts/update_streetview_cities.py
        # Format: (latitude, longitude, radius in degrees)
        high_density_locations = [
    (34.052234, -118.243685, 0.05),  # Los Angeles, Unknown
    (32.715738, -117.161084, 0.05),  # San Diego, Unknown
    (37.338208, -121.886329, 0.05),  # San Jose, Unknown
    (30.332184, -81.655651, 0.05),  # Jacksonville, Unknown
    (37.774929, -122.419415, 0.05),  # San Francisco, Unknown
    (39.961176, -82.998794, 0.05),  # Columbus, Unknown
    (42.331427, -83.045754, 0.05),  # Detroit, Unknown
    (47.606209, -122.332071, 0.05),  # Seattle, Unknown
    (35.467560, -97.516428, 0.05),  # Oklahoma City, Unknown
    (45.523062, -122.676482, 0.05),  # Portland, Unknown
    (43.038902, -87.906474, 0.05),  # Milwaukee, Unknown
    (36.746842, -119.772587, 0.05),  # Fresno, Unknown
    (38.581572, -121.494400, 0.05),  # Sacramento, Unknown
    (33.770050, -118.193740, 0.05),  # Long Beach, Unknown
    (39.099727, -94.578567, 0.05),  # Kansas City, Unknown
    (25.761680, -80.191790, 0.05),  # Miami, Unknown
    (37.804364, -122.271114, 0.05),  # Oakland, Unknown
    (44.977753, -93.265011, 0.05),  # Minneapolis, Unknown
    (36.153982, -95.992775, 0.05),  # Tulsa, Unknown
    (41.499320, -81.694361, 0.05),  # Cleveland, Unknown
    (35.373292, -119.018713, 0.05),  # Bakersfield, Unknown
    (27.950575, -82.457178, 0.05),  # Tampa, Unknown
    (33.835293, -117.914504, 0.05),  # Anaheim, Unknown
    (33.745573, -117.867834, 0.05),  # Santa Ana, Unknown
    (38.627003, -90.199404, 0.05),  # St. Louis, Unknown
    (33.953349, -117.396156, 0.05),  # Riverside, Unknown
    (37.957702, -121.290780, 0.05),  # Stockton, Unknown
    (39.103118, -84.512020, 0.05),  # Cincinnati, Unknown
    (44.953703, -93.089958, 0.05),  # St. Paul, Unknown
    (41.663938, -83.555212, 0.05),  # Toledo, Unknown
    (32.640054, -117.084196, 0.05),  # Chula Vista, Unknown
    (28.538335, -81.379237, 0.05),  # Orlando, Unknown
    (27.773056, -82.640000, 0.05),  # St. Petersburg, Unknown
    (43.073052, -89.401230, 0.05),  # Madison, Unknown
    (33.683947, -117.794694, 0.05),  # Irvine, Unknown
    (25.857596, -80.278106, 0.05),  # Hialeah, Unknown
    (37.548270, -121.988572, 0.05),  # Fremont, Unknown
    (34.108345, -117.289765, 0.05),  # San Bernardino, Unknown
    (47.658780, -117.426047, 0.05),  # Spokane, Unknown
    (37.639097, -120.996878, 0.05),  # Modesto, Unknown
    (47.252877, -122.444291, 0.05),  # Tacoma, Unknown
    (34.197505, -119.177052, 0.05),  # Oxnard, Unknown
    (34.092233, -117.435048, 0.05),  # Fontana, Unknown
    (33.942466, -117.229672, 0.05),  # Moreno Valley, Unknown
    (41.081445, -81.519005, 0.05),  # Akron, Unknown
    (33.660297, -117.999227, 0.05),  # Huntington Beach, Unknown
    (34.142508, -118.255075, 0.05),  # Glendale, Unknown
    (42.963360, -85.668086, 0.05),  # Grand Rapids, Unknown
    (40.760779, -111.891047, 0.05),  # Salt Lake City, Unknown
    (30.438256, -84.280733, 0.05),  # Tallahassee, Unknown
    (34.391664, -118.542586, 0.05),  # Santa Clarita, Unknown
    (33.773905, -117.941448, 0.05),  # Garden Grove, Unknown
    (33.195870, -117.379483, 0.05),  # Oceanside, Unknown
    (32.298757, -90.184810, 0.05),  # Jackson, Unknown
    (26.122439, -80.137317, 0.05),  # Fort Lauderdale, Unknown
    (38.440429, -122.714055, 0.05),  # Santa Rosa, Unknown
    (34.106399, -117.593108, 0.05),  # Rancho Cucamonga, Unknown
    (27.273049, -80.358226, 0.05),  # Port St. Lucie, Unknown
    (34.063344, -117.650888, 0.05),  # Ontario, Unknown
    (45.638728, -122.661486, 0.05),  # Vancouver, Unknown
    (26.562854, -81.949533, 0.05),  # Cape Coral, Unknown
    (37.208957, -93.292299, 0.05),  # Springfield, Unknown
    (26.007765, -80.296256, 0.05),  # Pembroke Pines, Unknown
    (38.408799, -121.371618, 0.05),  # Elk Grove, Unknown
    (44.942898, -123.035096, 0.05),  # Salem, Unknown
    (34.686785, -118.154163, 0.05),  # Lancaster, Unknown
    (33.875293, -117.566438, 0.05),  # Corona, Unknown
    (44.052069, -123.086754, 0.05),  # Eugene, Unknown
    (34.579434, -118.116461, 0.05),  # Palmdale, Unknown
    (36.677737, -121.655501, 0.05),  # Salinas, Unknown
    (37.668821, -122.080796, 0.05),  # Hayward, Unknown
    (34.055103, -117.749991, 0.05),  # Pomona, Unknown
    (33.119207, -117.086421, 0.05),  # Escondido, Unknown
    (37.368830, -122.036350, 0.05),  # Sunnyvale, Unknown
    (33.835849, -118.340629, 0.05),  # Torrance, Unknown
    (26.011201, -80.149490, 0.05),  # Hollywood, Unknown
    (39.758948, -84.191607, 0.05),  # Dayton, Unknown
    (33.787794, -117.853112, 0.05),  # Orange, Unknown
    (34.147785, -118.144515, 0.05),  # Pasadena, Unknown
    (33.870360, -117.924297, 0.05),  # Fullerton, Unknown
    (42.514457, -83.014653, 0.05),  # Warren, Unknown
    (47.610377, -122.200679, 0.05),  # Bellevue, Unknown
    (40.691613, -112.001050, 0.05),  # West Valley City, Unknown
    (42.580312, -83.030203, 0.05),  # Sterling Heights, Unknown
    (25.986076, -80.303560, 0.05),  # Miramar, Unknown
    (34.170561, -118.837594, 0.05),  # Thousand Oaks, Unknown
    (36.330228, -119.292058, 0.05),  # Visalia, Unknown
    (29.651634, -82.324826, 0.05),  # Gainesville, Unknown
    (38.752124, -121.288006, 0.05),  # Roseville, Unknown
    (26.271192, -80.270604, 0.05),  # Coral Springs, Unknown
    (34.269447, -118.781482, 0.05),  # Simi Valley, Unknown
    (37.977978, -122.031073, 0.05),  # Concord, Unknown
    (47.380933, -122.234843, 0.05),  # Kent, Unknown
    (34.536218, -117.292764, 0.05),  # Victorville, Unknown
    (37.354108, -121.955236, 0.05),  # Santa Clara, Unknown
    (38.104086, -122.256637, 0.05),  # Vallejo, Unknown
    (35.222567, -97.439478, 0.05),  # Norman, Unknown
    (39.091116, -94.415507, 0.05),  # Independence, Unknown
    (42.280826, -83.743038, 0.05),  # Ann Arbor, Unknown
    (37.871593, -122.272747, 0.05),  # Berkeley, Unknown
    (40.233844, -111.658534, 0.05),  # Provo, Unknown
    (34.068621, -118.027567, 0.05),  # El Monte, Unknown
    (38.951705, -92.334072, 0.05),  # Columbia, Unknown
    (42.732535, -84.555535, 0.05),  # Lansing, Unknown
    (33.940109, -118.133159, 0.05),  # Downey, Unknown
    (33.641132, -117.918669, 0.05),  # Costa Mesa, Unknown
    (33.961680, -118.353131, 0.05),  # Inglewood, Unknown
    (25.942038, -80.245604, 0.05),  # Miami Gardens, Unknown
    (33.158093, -117.350594, 0.05),  # Carlsbad, Unknown
    (44.012122, -92.480199, 0.05),  # Rochester, Unknown
    (40.609670, -111.939103, 0.05),  # West Jordan, Unknown
    (27.965853, -82.800103, 0.05),  # Clearwater, Unknown
    (45.500136, -122.430201, 0.05),  # Gresham, Unknown
    (38.249358, -122.039966, 0.05),  # Fairfield, Unknown
    (45.783286, -108.500690, 0.05),  # Billings, Unknown
    (34.274646, -119.229032, 0.05),  # San Buenaventura (Ventura), Unknown
    (34.068621, -117.938953, 0.05),  # West Covina, Unknown
    (37.935758, -122.347749, 0.05),  # Richmond, Unknown
    (33.553914, -117.213923, 0.05),  # Murrieta, Unknown
    (38.004921, -121.805789, 0.05),  # Antioch, Unknown
    (33.493639, -117.148365, 0.05),  # Temecula, Unknown
    (33.902237, -118.081733, 0.05),  # Norwalk, Unknown
    (47.978985, -122.202079, 0.05),  # Everett, Unknown
    (28.034462, -80.588665, 0.05),  # Palm Bay, Unknown
    (44.519159, -88.019826, 0.05),  # Green Bay, Unknown
    (37.687924, -122.470208, 0.05),  # Daly City, Unknown
    (34.180839, -118.308966, 0.05),  # Burbank, Unknown
    (26.237860, -80.124767, 0.05),  # Pompano Beach, Unknown
    (36.060949, -95.797453, 0.05),  # Broken Arrow, Unknown
    (26.715342, -80.053375, 0.05),  # West Palm Beach, Unknown
    (34.953034, -120.435719, 0.05),  # Santa Maria, Unknown
    (32.794773, -116.962527, 0.05),  # El Cajon, Unknown
    (34.106400, -117.370323, 0.05),  # Rialto, Unknown
    (37.562992, -122.325525, 0.05),  # San Mateo, Unknown
    (28.039465, -81.949804, 0.05),  # Lakeland, Unknown
    (35.227087, -80.843127, 0.05),  # Charlotte, Unknown
    (35.779590, -78.638179, 0.05),  # Raleigh, Unknown
    (61.218056, -149.900278, 0.05),  # Anchorage, Unknown
    (36.072635, -79.791975, 0.05),  # Greensboro, Unknown
    (35.994033, -78.898619, 0.05),  # Durham, Unknown
    (36.099860, -80.244216, 0.05),  # Winston-Salem, Unknown
    (33.520661, -86.802490, 0.05),  # Birmingham, Unknown
    (35.052664, -78.878358, 0.05),  # Fayetteville, Unknown
    (32.366805, -86.299969, 0.05),  # Montgomery, Unknown
    (30.695366, -88.039891, 0.05),  # Mobile, Unknown
    (34.730369, -86.586104, 0.05),  # Huntsville, Unknown
    (35.791540, -78.781117, 0.05),  # Cary, Unknown
    (46.877186, -96.789803, 0.05),  # Fargo, Unknown
    (34.225726, -77.944710, 0.05),  # Wilmington, Unknown
    (35.955692, -80.005318, 0.05),  # High Point, Unknown
    (41.878114, -87.629798, 0.05),  # Chicago, Unknown
    (39.768403, -86.158068, 0.05),  # Indianapolis, Unknown
    (41.079273, -85.139351, 0.05),  # Fort Wayne, Unknown
    (43.618710, -116.214607, 0.05),  # Boise City, Unknown
    (41.760585, -88.320072, 0.05),  # Aurora, Unknown
    (42.271131, -89.093995, 0.05),  # Rockford, Unknown
    (41.525031, -88.081725, 0.05),  # Joliet, Unknown
    (41.750839, -88.153535, 0.05),  # Naperville, Unknown
    (37.971559, -87.571090, 0.05),  # Evansville, Unknown
    (39.781721, -89.650148, 0.05),  # Springfield, Unknown
    (40.693649, -89.588986, 0.05),  # Peoria, Unknown
    (42.035408, -88.282567, 0.05),  # Elgin, Unknown
    (41.676355, -86.251990, 0.05),  # South Bend, Unknown
    (40.712784, -74.005941, 0.05),  # New York, Unknown
    (42.360082, -71.058880, 0.05),  # Boston, Unknown
    (39.290385, -76.612189, 0.05),  # Baltimore, Unknown
    (38.252665, -85.758456, 0.05),  # Louisville/Jefferson County, Unknown
    (36.169941, -115.139830, 0.05),  # Las Vegas, Unknown
    (35.085334, -106.605553, 0.05),  # Albuquerque, Unknown
    (41.252363, -95.997988, 0.05),  # Omaha, Unknown
    (38.040584, -84.503716, 0.05),  # Lexington-Fayette, Unknown
    (40.735657, -74.172367, 0.05),  # Newark, Unknown
    (36.039525, -114.981721, 0.05),  # Henderson, Unknown
    (40.825763, -96.685198, 0.05),  # Lincoln, Unknown
    (42.886447, -78.878369, 0.05),  # Buffalo, Unknown
    (40.728157, -74.077642, 0.05),  # Jersey City, Unknown
    (39.529633, -119.813803, 0.05),  # Reno, Unknown
    (36.198859, -115.117501, 0.05),  # North Las Vegas, Unknown
    (43.161030, -77.610922, 0.05),  # Rochester, Unknown
    (40.931210, -73.898747, 0.05),  # Yonkers, Unknown
    (42.262593, -71.802293, 0.05),  # Worcester, Unknown
    (43.544596, -96.731103, 0.05),  # Sioux Falls, Unknown
    (42.101483, -72.589811, 0.05),  # Springfield, Unknown
    (40.916765, -74.171811, 0.05),  # Paterson, Unknown
    (43.048122, -76.147424, 0.05),  # Syracuse, Unknown
    (34.000710, -81.034814, 0.05),  # Columbia, Unknown
    (32.776475, -79.931051, 0.05),  # Charleston, Unknown
    (40.663992, -74.210701, 0.05),  # Elizabeth, Unknown
    (42.995640, -71.454789, 0.05),  # Manchester, Unknown
    (42.633425, -71.316172, 0.05),  # Lowell, Unknown
    (42.373616, -71.109733, 0.05),  # Cambridge, Unknown
    (32.854620, -79.974810, 0.05),  # North Charleston, Unknown
    (32.319940, -106.763654, 0.05),  # Las Cruces, Unknown
    (39.952584, -75.165222, 0.05),  # Philadelphia, Unknown
    (33.448377, -112.074037, 0.05),  # Phoenix, Unknown
    (39.739236, -104.990251, 0.05),  # Denver, Unknown
    (32.221743, -110.926479, 0.05),  # Tucson, Unknown
    (33.415184, -111.831472, 0.05),  # Mesa, Unknown
    (38.833882, -104.821363, 0.05),  # Colorado Springs, Unknown
    (39.729432, -104.831919, 0.05),  # Aurora, Unknown
    (40.440625, -79.995886, 0.05),  # Pittsburgh, Unknown
    (33.306160, -111.841250, 0.05),  # Chandler, Unknown
    (33.538652, -112.185987, 0.05),  # Glendale, Unknown
    (33.352826, -111.789027, 0.05),  # Gilbert, Unknown
    (33.494170, -111.926052, 0.05),  # Scottsdale, Unknown
    (34.746481, -92.289595, 0.05),  # Little Rock, Unknown
    (33.425510, -111.940005, 0.05),  # Tempe, Unknown
    (33.580596, -112.237378, 0.05),  # Peoria, Unknown
    (40.585260, -105.084423, 0.05),  # Fort Collins, Unknown
    (41.186548, -73.195177, 0.05),  # Bridgeport, Unknown
    (39.704709, -105.081373, 0.05),  # Lakewood, Unknown
    (41.308274, -72.927883, 0.05),  # New Haven, Unknown
    (39.868041, -104.971924, 0.05),  # Thornton, Unknown
    (41.053430, -73.538734, 0.05),  # Stamford, Unknown
    (41.763711, -72.685093, 0.05),  # Hartford, Unknown
    (33.629234, -112.367928, 0.05),  # Surprise, Unknown
    (40.608430, -75.490183, 0.05),  # Allentown, Unknown
    (39.802764, -105.087484, 0.05),  # Arvada, Unknown
    (39.836653, -105.037205, 0.05),  # Westminster, Unknown
    (41.558152, -73.051496, 0.05),  # Waterbury, Unknown
    (38.254447, -104.609141, 0.05),  # Pueblo, Unknown
    (39.580745, -104.877173, 0.05),  # Centennial, Unknown
    (40.014986, -105.270546, 0.05),  # Boulder, Unknown
    (42.129224, -80.085059, 0.05),  # Erie, Unknown
]
        
        # Use successful locations cache if we have any
        if len(StreetViewClient._successful_locations) > 0 and random.random() < 0.3:
            # 30% chance to use a previously successful location with small variation
            cache_location = random.choice(StreetViewClient._successful_locations)
            base_lat, base_lon = cache_location
            
            # Add small variation (within ~500m)
            variation = 0.005  # ~500m in degrees
            lat_offset = random.uniform(-variation, variation)
            lon_offset = random.uniform(-variation, variation)
            
            latitude = base_lat + lat_offset
            longitude = base_lon + lon_offset
            
            # Ensure within original bounds
            latitude = max(min_lat, min(max_lat, latitude))
            longitude = max(min_lon, min(max_lon, longitude))
            
            return latitude, longitude
        
        # Strategy selection with heavy bias toward urban centers (90%)
        strategy = random.choices(
            ['urban_center', 'global'], 
            weights=[0.9, 0.1],
            k=1
        )[0]
        
        if strategy == 'urban_center':
            # Choose a random urban center
            center = random.choice(high_density_locations)
            center_lat, center_lon, radius = center
            
            # Check if the center is within the requested hemisphere bounds
            if not (min_lat <= center_lat <= max_lat):
                # Outside bounds, try another approach
                return self.get_random_coordinates(min_lat, max_lat, min_lon, max_lon)
            
            # Generate point within radius of center
            # Using a more uniform distribution within the circle
            r = radius * math.sqrt(random.random())  # sqrt for uniform distribution in circle
            theta = random.uniform(0, 2 * math.pi)
            
            # Convert polar to cartesian, approximating degrees
            lat_offset = r * math.cos(theta)
            lon_offset = r * math.sin(theta) / math.cos(math.radians(center_lat))
            
            latitude = center_lat + lat_offset
            longitude = center_lon + lon_offset
            
        else:  # global
            # Completely random within provided bounds
            latitude = random.uniform(min_lat, max_lat)
            longitude = random.uniform(min_lon, max_lon)
            
        # Ensure within original bounds (in case of any math errors)
        latitude = max(min_lat, min(max_lat, latitude))
        longitude = max(min_lon, min(max_lon, longitude))
            
        return latitude, longitude
    
    def record_successful_location(self, latitude: float, longitude: float) -> None:
        """Record a location that successfully had a panorama.
        
        Args:
            latitude: Latitude of successful location
            longitude: Longitude of successful location
        """
        if len(StreetViewClient._successful_locations) >= StreetViewClient._max_cache_size:
            # Remove random item to keep cache size manageable
            if random.random() < 0.5:  # 50% chance to remove
                StreetViewClient._successful_locations.pop(
                    random.randrange(len(StreetViewClient._successful_locations))
                )
        
        # Add new successful location
        StreetViewClient._successful_locations.append((latitude, longitude))
    
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
            print(f"API Error: Status code {response.status_code}, Response: {response.text}")
            return False
        
        data = response.json()
        if data.get("status") != "OK":
            print(f"API Error: Status {data.get('status')}, Error message: {data.get('error_message', 'None')}")
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
        
        # Record this as a successful location for future use
        self.record_successful_location(latitude, longitude)
        
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
        max_attempts: int = 20,  # Increased from 10 to 20
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
                
            # Minimal delay for speed
            time.sleep(0.01)
            
        return None, None