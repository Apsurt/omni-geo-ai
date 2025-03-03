#!/usr/bin/env python3
"""
Script to automatically update the StreetViewClient with a generated list of ~1000 cities.

This script:
1. Runs the generate_city_list.py script to create a city list
2. Updates the StreetViewClient.py file with the new city list
"""
import os
import sys
import subprocess
import re

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_city_list():
    """Run the generate_city_list.py script to create the city list."""
    print("Generating city list...")
    
    # Check if the script exists
    script_path = os.path.join(os.path.dirname(__file__), 'generate_city_list.py')
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return False
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Run the script
    try:
        subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running generate_city_list.py: {e}")
        return False


def read_city_list():
    """Read the generated city list file."""
    # Try in the scripts directory first
    city_list_path = os.path.join(os.path.dirname(__file__), 'city_list.py')
    
    # If not found, try in the project root (where generate_city_list.py is actually saving it)
    if not os.path.exists(city_list_path):
        city_list_path = os.path.join(os.path.dirname(__file__), '..', 'city_list.py')
    
    if not os.path.exists(city_list_path):
        print(f"Error: city_list.py not found in either scripts directory or project root")
        return None
    
    print(f"Found city list at: {city_list_path}")
    
    with open(city_list_path, 'r') as f:
        content = f.read()
    
    # Extract the list part
    match = re.search(r'HIGH_DENSITY_LOCATIONS = \[(.*?)\]', content, re.DOTALL)
    if not match:
        print("Error: Could not find HIGH_DENSITY_LOCATIONS in generated file")
        return None
    
    return match.group(1)


def update_streetview_client(city_list_content):
    """Update the StreetViewClient.py file with the new city list."""
    sv_client_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'omni_geo_ai', 
        'data_collection', 
        'streetview_client.py'
    )
    
    if not os.path.exists(sv_client_path):
        print(f"Error: {sv_client_path} not found")
        return False
    
    # Read the StreetViewClient.py file
    with open(sv_client_path, 'r') as f:
        content = f.read()
    
    # Find the existing high_density_locations list
    # Different variable name casing in the actual file
    patterns = [
        r'high_density_locations = \[(.*?)\]',
        r'HIGH_DENSITY_LOCATIONS = \[(.*?)\]', 
        r'high_density_locations\s*=\s*\[(.*?)\]'
    ]
    
    match = None
    pattern_used = None
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            pattern_used = pattern
            break
    
    if not match:
        # Try to find any list that might be the cities
        print("Could not find high_density_locations with standard patterns")
        # Look for a list of tuples with city coordinates
        pattern = r'(\[\s*#[\s\S]*?[\(][-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+[\)][\s\S]*?\])'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            # Replace the entire list section
            list_section = match.group(1)
            print(f"Found a list section to replace with coordinates")
            new_content = content.replace(list_section, f"[\n        {city_list_content}\n    ]")
            
            # Write the updated file
            with open(sv_client_path, 'w') as f:
                f.write(new_content)
            
            print(f"Successfully updated {sv_client_path} with new city list")
            return True
        else:
            print("Error: Could not find any city list in StreetViewClient.py")
            return False
    
    # Replace the list with our new one
    new_content = re.sub(
        pattern_used,
        f'high_density_locations = [{city_list_content}]',
        content,
        flags=re.DOTALL
    )
    
    # Write the updated file
    with open(sv_client_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully updated {sv_client_path} with new city list")
    return True


def cleanup():
    """Remove the temporary city_list.py file."""
    # Try both possible locations
    locations = [
        os.path.join(os.path.dirname(__file__), 'city_list.py'),
        os.path.join(os.path.dirname(__file__), '..', 'city_list.py')
    ]
    
    for city_list_path in locations:
        if os.path.exists(city_list_path):
            os.remove(city_list_path)
            print(f"Removed temporary file {city_list_path}")


def main():
    """Main function to update the StreetViewClient with ~1000 cities."""
    print("Updating StreetViewClient with ~1000 cities...")
    
    # Generate the city list
    if not generate_city_list():
        print("Failed to generate city list. Exiting.")
        return
    
    # Read the generated city list
    city_list_content = read_city_list()
    if not city_list_content:
        print("Failed to read city list. Exiting.")
        return
    
    # Update the StreetViewClient
    if update_streetview_client(city_list_content):
        print("StreetViewClient updated successfully with ~1000 cities!")
    else:
        print("Failed to update StreetViewClient.")
    
    # Cleanup temporary files
    cleanup()


if __name__ == "__main__":
    main()