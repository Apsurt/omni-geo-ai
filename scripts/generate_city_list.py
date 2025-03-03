#!/usr/bin/env python3
"""
Script to generate an extended list of cities for the StreetViewClient.

This script will:
1. Download open data for world cities
2. Filter to select ~1000 cities based on population and region diversity
3. Format them as entries for the StreetViewClient._high_density_locations list
"""
import os
import sys
import pandas as pd
import numpy as np
import requests
from io import StringIO
import json
from tqdm import tqdm

# Add parent directory to path to import project modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Data source: https://simplemaps.com/data/world-cities (free basic database)
CITIES_CSV_URL = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.75.csv"

# Alternative data sources if above doesn't work
# Try these in order until one works
BACKUP_URLS = [
    # This one has lat, lng, population and name
    "https://gist.githubusercontent.com/Miserlou/c5cd8364bf9b2420bb29/raw/2bf258763cdddd704f8ffd3ea9a3e81d25e2c6f6/cities.json",
    # These are additional options
    "https://gist.githubusercontent.com/curran/13d30e855d48cdd6f22acdf0afe27286/raw/0635f14817ec634833bb904a47594cc2f5f9dbf8/worldcities_clean.csv",
    "https://raw.githubusercontent.com/lutangar/cities.json/master/cities.json"
]

# Create a basic hardcoded city list in case all else fails
FALLBACK_CITIES = """lat,lng,city,country,population
40.7128,-74.0060,New York,United States,8400000
34.0522,-118.2437,Los Angeles,United States,4000000
41.8781,-87.6298,Chicago,United States,2700000
51.5074,-0.1278,London,United Kingdom,8900000
48.8566,2.3522,Paris,France,2100000
"""

# Target number of cities
TARGET_CITIES = 1000

# Minimum population to consider
MIN_POPULATION = 100000

# Radius in degrees around each city (can be adjusted)
DEFAULT_RADIUS = 0.05

# Continent diversity weights
CONTINENT_WEIGHTS = {
    "North America": 0.2,
    "Europe": 0.25,
    "Asia": 0.25,
    "Africa": 0.15,
    "South America": 0.1,
    "Oceania": 0.05
}

# Mapping country codes to continents
COUNTRY_TO_CONTINENT = {
    # North America
    "US": "North America", "CA": "North America", "MX": "North America", 
    "PR": "North America", "DO": "North America", "HT": "North America", 
    "CU": "North America", "GT": "North America", "HN": "North America",
    "SV": "North America", "NI": "North America", "CR": "North America", 
    "PA": "North America", "BS": "North America", "JM": "North America",
    "BZ": "North America", "AL": "North America", "AK": "North America",
    "AZ": "North America", "AR": "North America", "CA": "North America",
    "CO": "North America", "CT": "North America", "DE": "North America",
    "FL": "North America", "GA": "North America", "HI": "North America",
    "ID": "North America", "IL": "North America", "IN": "North America",
    "IA": "North America", "KS": "North America", "KY": "North America",
    "LA": "North America", "ME": "North America", "MD": "North America",
    "MA": "North America", "MI": "North America", "MN": "North America",
    "MS": "North America", "MO": "North America", "MT": "North America",
    "NE": "North America", "NV": "North America", "NH": "North America",
    "NJ": "North America", "NM": "North America", "NY": "North America",
    "NC": "North America", "ND": "North America", "OH": "North America",
    "OK": "North America", "OR": "North America", "PA": "North America",
    "RI": "North America", "SC": "North America", "SD": "North America",
    "TN": "North America", "TX": "North America", "UT": "North America",
    "VT": "North America", "VA": "North America", "WA": "North America",
    "WV": "North America", "WI": "North America", "WY": "North America",
    
    # Europe
    "GB": "Europe", "DE": "Europe", "FR": "Europe", "ES": "Europe", "IT": "Europe", 
    "PT": "Europe", "CH": "Europe", "AT": "Europe", "BE": "Europe", "NL": "Europe",
    "DK": "Europe", "SE": "Europe", "NO": "Europe", "FI": "Europe", "PL": "Europe",
    "CZ": "Europe", "SK": "Europe", "HU": "Europe", "RO": "Europe", "BG": "Europe",
    "GR": "Europe", "TR": "Europe", "RU": "Europe", "UA": "Europe", "IE": "Europe",
    "IS": "Europe", "LT": "Europe", "LV": "Europe", "EE": "Europe", "BY": "Europe",
    "MD": "Europe", "AL": "Europe", "HR": "Europe", "BA": "Europe", "RS": "Europe",
    "ME": "Europe", "MK": "Europe", "SI": "Europe", "CY": "Europe", "MT": "Europe", 
    "LU": "Europe", "LI": "Europe", "MC": "Europe", "SM": "Europe", "VA": "Europe",
    "AD": "Europe", "UK": "Europe",
    
    # Asia
    "CN": "Asia", "JP": "Asia", "IN": "Asia", "KR": "Asia", "ID": "Asia", 
    "MY": "Asia", "SG": "Asia", "TH": "Asia", "VN": "Asia", "PH": "Asia",
    "HK": "Asia", "TW": "Asia", "LK": "Asia", "BD": "Asia", "PK": "Asia",
    "NP": "Asia", "BT": "Asia", "MM": "Asia", "LA": "Asia", "KH": "Asia",
    "MN": "Asia", "BN": "Asia", "KZ": "Asia", "KG": "Asia", "TJ": "Asia",
    "TM": "Asia", "UZ": "Asia", "AF": "Asia", "IQ": "Asia", "IR": "Asia",
    "SA": "Asia", "YE": "Asia", "SY": "Asia", "JO": "Asia", "IL": "Asia",
    "LB": "Asia", "AE": "Asia", "QA": "Asia", "BH": "Asia", "KW": "Asia",
    "OM": "Asia",
    
    # South America
    "BR": "South America", "AR": "South America", "CL": "South America", 
    "CO": "South America", "PE": "South America", "VE": "South America",
    "EC": "South America", "BO": "South America", "PY": "South America",
    "UY": "South America", "GY": "South America", "SR": "South America",
    "GF": "South America",
    
    # Oceania
    "AU": "Oceania", "NZ": "Oceania", "FJ": "Oceania", "PG": "Oceania",
    "SB": "Oceania", "VU": "Oceania", "NC": "Oceania", "PF": "Oceania",
    "WS": "Oceania", "TO": "Oceania", "TV": "Oceania", "KI": "Oceania",
    "MH": "Oceania", "FM": "Oceania", "PW": "Oceania", "NR": "Oceania",
    
    # Africa
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "KE": "Africa", "MA": "Africa",
    "TN": "Africa", "DZ": "Africa", "GH": "Africa", "ET": "Africa", "CD": "Africa",
    "TZ": "Africa", "UG": "Africa", "ZM": "Africa", "ZW": "Africa", "AO": "Africa",
    "NA": "Africa", "MZ": "Africa", "MG": "Africa", "CI": "Africa", "CM": "Africa",
    "SN": "Africa", "SO": "Africa", "ML": "Africa", "NE": "Africa", "TD": "Africa",
    "SD": "Africa", "SS": "Africa", "BF": "Africa", "GN": "Africa", "BJ": "Africa",
    "LR": "Africa", "SL": "Africa", "TG": "Africa", "GM": "Africa", "CV": "Africa",
    "CG": "Africa", "GA": "Africa", "GQ": "Africa", "MW": "Africa", "RW": "Africa",
    "BI": "Africa", "ER": "Africa", "DJ": "Africa", "SZ": "Africa", "LS": "Africa",
    "BW": "Africa", "LY": "Africa", "MR": "Africa"
}

# Add more country codes as needed to the mapping


def download_cities_data():
    """Download the cities dataset."""
    try:
        print(f"Downloading cities data from {CITIES_CSV_URL}...")
        response = requests.get(CITIES_CSV_URL)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.text, "primary"
    except Exception as e:
        print(f"Error downloading from primary source: {e}")
        
        # Try backup URLs in order
        for i, url in enumerate(BACKUP_URLS):
            try:
                print(f"Trying backup URL {i+1}: {url}")
                response = requests.get(url)
                response.raise_for_status()
                
                # Check if this is a JSON source (not CSV)
                if url.endswith('.json'):
                    import json
                    data = json.loads(response.text)
                    # Convert JSON to CSV
                    import pandas as pd
                    df = pd.DataFrame(data)
                    csv_data = df.to_csv(index=False)
                    return csv_data, f"backup{i+1}_json"
                
                return response.text, f"backup{i+1}"
            except Exception as backup_error:
                print(f"Error downloading from backup source {i+1}: {backup_error}")
        
        # If we reach here, all sources failed
        print("All download attempts failed, using fallback hardcoded list")
        return FALLBACK_CITIES, "fallback"


def get_continent(country_code):
    """Map a country code to its continent."""
    return COUNTRY_TO_CONTINENT.get(country_code, "Unknown")


def stratified_selection(df, target_count):
    """Select cities with stratification by continent."""
    # Ensure we have a continent column
    if 'continent' not in df.columns:
        df['continent'] = df['iso2'].apply(get_continent)
    
    # Count cities by continent
    continent_counts = df['continent'].value_counts().to_dict()
    print(f"Cities by continent: {continent_counts}")
    
    # Set default continent for missing data
    if 'Unknown' in continent_counts and continent_counts['Unknown'] > 0:
        print(f"WARNING: {continent_counts['Unknown']} cities with unknown continent")
        
        # Check if we have state information for US cities
        if 'state' in df.columns:
            print("Using state information to identify US cities")
            us_states = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
            ]
            # Identify cities in US states
            mask = df['continent'] == 'Unknown'
            for idx, row in df[mask].iterrows():
                if isinstance(row['state'], str) and row['state'].upper() in us_states:
                    df.at[idx, 'continent'] = 'North America'
                    
            # Check again how many are still unknown
            unknown_count = (df['continent'] == 'Unknown').sum()
            print(f"After state processing: {unknown_count} cities still have unknown continent")
            
        if (df['continent'] == 'Unknown').sum() > 0.8 * df.shape[0]:
            # Most cities have unknown continent, probably US cities only
            print("Assuming most cities are from North America")
            df.loc[df['continent'] == 'Unknown', 'continent'] = 'North America'
    
    # Adjust weights based on available data
    adjusted_weights = {}
    total_weight = 0
    
    for continent, weight in CONTINENT_WEIGHTS.items():
        if continent in continent_counts and continent_counts[continent] > 0:
            adjusted_weights[continent] = weight
            total_weight += weight
    
    # Normalize weights to sum to 1
    if total_weight > 0:
        for continent in adjusted_weights:
            adjusted_weights[continent] = adjusted_weights[continent] / total_weight
    else:
        # Fallback to one continent if no data matched
        main_continent = next(iter(continent_counts.keys()))
        adjusted_weights = {main_continent: 1.0}
    
    print(f"Adjusted continent weights: {adjusted_weights}")
    
    # Calculate targets per continent
    continent_targets = {}
    for continent, weight in adjusted_weights.items():
        continent_targets[continent] = max(1, int(target_count * weight))
    
    # Add remaining cities to largest continents
    remaining = target_count - sum(continent_targets.values())
    for continent in sorted(adjusted_weights, key=adjusted_weights.get, reverse=True):
        if remaining <= 0:
            break
        continent_targets[continent] += 1
        remaining -= 1
    
    print(f"Target cities per continent: {continent_targets}")
    
    # Select cities from each continent
    selected_cities = []
    for continent, target in continent_targets.items():
        continent_df = df[df['continent'] == continent]
        
        if len(continent_df) <= target:
            # Take all cities if we have fewer than the target
            selected_cities.append(continent_df)
            print(f"Selected all {len(continent_df)} cities from {continent}")
        else:
            # Otherwise take the largest cities
            largest = continent_df.nlargest(target, 'population')
            selected_cities.append(largest)
            print(f"Selected {len(largest)} largest cities from {continent}")
    
    # Combine selections
    result = pd.concat(selected_cities)
    print(f"Total cities selected: {len(result)}")
    return result


def format_city_entries(selected_df):
    """Format the DataFrame into Python code for the StreetViewClient."""
    city_entries = []
    
    for _, row in selected_df.iterrows():
        try:
            city_name = row.get('city', row.get('name', 'Unknown'))
            country_name = row.get('country', 'Unknown')
            
            entry = (
                f"({row['lat']:.6f}, {row['lng']:.6f}, {DEFAULT_RADIUS}),  "
                f"# {city_name}, {country_name}"
            )
            city_entries.append(entry)
        except Exception as e:
            print(f"Error formatting city entry: {e}")
            print(f"Problematic row: {row}")
            continue
    
    return city_entries


def main():
    """Main function to generate the city list."""
    # Download data
    csv_data, source_type = download_cities_data()
    if not csv_data:
        print("Failed to download data. Exiting.")
        return
    
    print(f"Successfully downloaded data from {source_type} source")
    
    # Load into pandas
    df = pd.read_csv(StringIO(csv_data))
    
    # Print available columns to help debug
    print(f"Available columns: {list(df.columns)}")
    
    # Handle different data sources with different column names
    print(f"Processing {source_type} source")
    
    # Convert any backup json columns to expected format
    if "_json" in source_type:
        # JSON dataset from backup
        print("Processing JSON dataset")
        
        # Check for common JSON keys and map to our standard columns
        key_mappings = {
            'lat': ['lat', 'latitude'],
            'lng': ['lng', 'lon', 'longitude'],
            'population': ['population', 'pop', 'populationCount'],
            'iso2': ['country_code', 'countryCode', 'iso2'],
            'city': ['city', 'name', 'cityName']
        }
        
        for target_col, possible_cols in key_mappings.items():
            for possible_col in possible_cols:
                if possible_col in df.columns and target_col not in df.columns:
                    df[target_col] = df[possible_col]
                    print(f"Mapped {possible_col} to {target_col}")
                    break
    
    # Handle specific dataset types
    elif source_type == "primary":
        # Handle primary dataset
        if 'pop' in df.columns:
            df['population'] = df['pop']
    
    elif source_type.startswith("backup"):
        # Any backup dataset
        # Renaming columns to match our expected format
        if 'lat' not in df.columns and 'latitude' in df.columns:
            df['lat'] = df['latitude']
        
        if 'lng' not in df.columns and 'longitude' in df.columns:
            df['lng'] = df['longitude']
            
        if 'population' not in df.columns and 'pop' in df.columns:
            df['population'] = df['pop']
            
        # If there's no city column, try to find one
        if 'city' not in df.columns:
            for city_col in ['name', 'city_name', 'city', 'cityName']:
                if city_col in df.columns:
                    df['city'] = df[city_col]
                    break
            
        # Handle missing population
        if 'population' not in df.columns:
            print("WARNING: No population data found, using dummy values")
            df['population'] = 1000000
    
    # Apply common transformations for all sources
    
    # Ensure we have the needed columns
    if 'lat' not in df.columns:
        print("ERROR: No latitude column found. Exiting.")
        return
        
    if 'lng' not in df.columns:
        print("ERROR: No longitude column found. Exiting.")
        return
    
    # Handle country codes
    if 'iso2' not in df.columns:
        if 'country_code' in df.columns:
            df['iso2'] = df['country_code']
        elif 'iso3' in df.columns:
            df['iso2'] = df['iso3'].apply(lambda x: x[:2] if isinstance(x, str) and len(x) >= 2 else 'XX')
        elif 'country' in df.columns:
            # Create a dummy iso2 from country name if needed
            df['iso2'] = df['country'].apply(lambda x: x[:2].upper() if isinstance(x, str) else 'XX')
            print("WARNING: Created dummy country codes from country names.")
        else:
            # Last resort: Use state or just assign a default
            if 'state' in df.columns:
                print("WARNING: Using state as substitute for country")
                df['iso2'] = df['state'].apply(lambda x: x[:2].upper() if isinstance(x, str) else 'XX')
            else:
                print("WARNING: No country information found, using default country code US")
                df['iso2'] = 'US'  # Default to US if no country info
    
    # Ensure we have a population column
    if 'population' not in df.columns:
        print("WARNING: No population data found, using dummy values")
        df['population'] = 1000000
    
    # Convert population to numeric (some datasets have it as string)
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df['population'].fillna(MIN_POPULATION, inplace=True)
    
    # Basic data cleaning
    df = df[df['population'] >= MIN_POPULATION]
    
    # Convert lat/lng to numeric and drop invalid rows
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])
    
    # Select cities with continental diversity
    selected_cities = stratified_selection(df, TARGET_CITIES)
    
    # Format as Python code
    city_entries = format_city_entries(selected_cities)
    
    # Output
    print(f"Generated {len(city_entries)} city entries.")
    
    # Write to file
    output_file = "city_list.py"
    with open(output_file, 'w') as f:
        f.write("# Auto-generated city list for StreetViewClient\n")
        f.write("HIGH_DENSITY_LOCATIONS = [\n")
        for entry in city_entries:
            f.write(f"    {entry}\n")
        f.write("]\n")
    
    print(f"City list written to {output_file}")
    print("Copy this list to StreetViewClient.py to replace the existing high_density_locations.")


if __name__ == "__main__":
    main()