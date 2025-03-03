#!/usr/bin/env python3
"""
Script to fetch Street View panoramas for Omni Geo AI.

This script can be run manually or set up as a cron job to collect
panorama images automatically on a daily basis.
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omni_geo_ai.config import DAILY_PANORAMA_LIMIT
from omni_geo_ai.database import init_db
from omni_geo_ai.data_collection.collector import PanoramaCollector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch Street View panoramas for Omni Geo AI")
    
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=DAILY_PANORAMA_LIMIT,
        help=f"Number of images to collect (default: {DAILY_PANORAMA_LIMIT})"
    )
    
    parser.add_argument(
        "--north_ratio", 
        type=float, 
        default=0.5,
        help="Target ratio of Northern hemisphere images (default: 0.5)"
    )
    
    parser.add_argument(
        "--output_stats", 
        type=str, 
        help="Optional path to save collection statistics as JSON"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Initialize database and validate settings without collecting images"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all panorama data from database and GCS storage before collecting"
    )
    
    parser.add_argument(
        "--clear_only",
        action="store_true",
        help="Clear all panorama data from database and GCS storage and exit without collecting"
    )
    
    return parser.parse_args()


def clear_data():
    """Clear all panorama data from database and GCS storage."""
    from omni_geo_ai.database import clear_db
    from omni_geo_ai.storage import StorageClient
    
    print("Clearing all panorama data...")
    
    # Clear database
    try:
        clear_db()
        print("✅ Database cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        return False
    
    # Clear GCS storage
    try:
        storage_client = StorageClient()
        deleted_count = storage_client.clear_all_panoramas()
        print(f"✅ Google Cloud Storage cleared successfully ({deleted_count} files deleted)")
    except Exception as e:
        print(f"❌ Error clearing Google Cloud Storage: {e}")
        return False
    
    return True


def main():
    """Main entry point for panorama fetching script."""
    args = parse_args()
    
    # Configure logging based on verbosity
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize database (create tables if they don't exist)
    init_db()
    
    # Handle clear_only flag (clear and exit)
    if args.clear_only:
        if clear_data():
            print("All data cleared successfully. Exiting.")
        else:
            print("Error clearing data. Some operations may have failed.")
        return
    
    # Handle clear flag (clear before continuing)
    if args.clear:
        if not clear_data():
            print("Error clearing data. Aborting collection.")
            return
    
    if args.dry_run:
        print("Dry run mode: Database initialized successfully")
        print("Configuration validated. Exiting without collecting images.")
        return
    
    # Initialize collector
    collector = PanoramaCollector()
    
    # Start collection
    print(f"Starting collection of {args.num_images} panoramas...")
    print(f"Target ratio: {args.north_ratio*100:.1f}% Northern / {(1-args.north_ratio)*100:.1f}% Southern")
    
    start_time = datetime.now()
    stats = collector.collect_panoramas(
        num_images=args.num_images,
        north_ratio=args.north_ratio
    )
    end_time = datetime.now()
    
    # Print summary
    elapsed = (end_time - start_time).total_seconds()
    print("\nCollection Summary:")
    print(f"- Requested: {stats['total_requested']} panoramas")
    print(f"- Successfully collected: {stats['successfully_stored']} panoramas")
    print(f"- Northern hemisphere: {stats['north_count']} panoramas")
    print(f"- Southern hemisphere: {stats['south_count']} panoramas")
    print(f"- Duplicates skipped: {stats['duplicates']}")
    print(f"- Failed attempts: {stats['failed']}")
    print(f"- Total time: {elapsed:.1f} seconds")
    
    if stats['by_country']:
        print("\nTop countries:")
        top_countries = sorted(
            stats['by_country'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for country, count in top_countries:
            print(f"- {country}: {count} panoramas")
    
    # Save statistics if requested
    if args.output_stats:
        with open(args.output_stats, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            stats['start_time'] = stats['start_time'].isoformat()
            stats['end_time'] = stats['end_time'].isoformat()
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {args.output_stats}")


if __name__ == "__main__":
    main()