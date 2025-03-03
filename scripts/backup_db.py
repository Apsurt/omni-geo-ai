#!/usr/bin/env python3
"""
Script to backup the Omni Geo AI PostgreSQL database to Google Cloud Storage.

This script should be set up as a daily cron job to ensure regular backups
of the database.
"""
import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.cloud import storage

from omni_geo_ai.config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    GCS_BUCKET_NAME
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Backup Omni Geo AI PostgreSQL database to GCS")
    
    parser.add_argument(
        "--bucket",
        type=str,
        default=GCS_BUCKET_NAME,
        help=f"GCS bucket name (default: {GCS_BUCKET_NAME})"
    )
    
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="backups",
        help="GCS directory for backups (default: 'backups')"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def backup_db_to_file(output_file):
    """Create a PostgreSQL dump backup file.
    
    Args:
        output_file: Path to save the backup file
        
    Returns:
        True if backup was successful, False otherwise
    """
    env = os.environ.copy()
    
    # Set PostgreSQL environment variables
    if DB_PASSWORD:
        env["PGPASSWORD"] = DB_PASSWORD
        
    # Build pg_dump command
    cmd = [
        "pg_dump",
        f"--host={DB_HOST}",
        f"--port={DB_PORT}",
        f"--username={DB_USER}",
        "--format=custom",      # Custom format for maximum compression and flexibility
        f"--file={output_file}",
        DB_NAME
    ]
    
    try:
        # Run pg_dump
        logging.info(f"Running pg_dump for database {DB_NAME}")
        process = subprocess.run(cmd, env=env, check=True, capture_output=True)
        
        # Verify the backup file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logging.info(f"Backup file created: {output_file} ({os.path.getsize(output_file)} bytes)")
            return True
        else:
            logging.error("Backup file not created or is empty")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"pg_dump failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr.decode('utf-8')}")
        return False
        
    except Exception as e:
        logging.error(f"Error during database backup: {e}")
        return False


def upload_to_gcs(local_file, bucket_name, gcs_path):
    """Upload a file to Google Cloud Storage.
    
    Args:
        local_file: Path to the local file to upload
        bucket_name: GCS bucket name
        gcs_path: Path within the bucket to store the file
        
    Returns:
        True if upload was successful, False otherwise
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        logging.info(f"Uploading {local_file} to gs://{bucket_name}/{gcs_path}")
        blob.upload_from_filename(local_file)
        
        logging.info(f"Upload complete: gs://{bucket_name}/{gcs_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        return False


def main():
    """Main entry point for database backup script."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create timestamped filename for the backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{DB_NAME}_{timestamp}.backup"
    local_backup_path = os.path.join("/tmp", backup_filename)
    
    # Create the backup file
    if not backup_db_to_file(local_backup_path):
        logging.error("Database backup failed")
        sys.exit(1)
    
    # Upload to GCS
    gcs_path = f"{args.backup_dir}/{backup_filename}"
    if upload_to_gcs(local_backup_path, args.bucket, gcs_path):
        print(f"Database successfully backed up to gs://{args.bucket}/{gcs_path}")
    else:
        logging.error("Failed to upload backup to GCS")
        sys.exit(1)
    
    # Clean up the local backup file
    try:
        os.remove(local_backup_path)
        logging.info(f"Removed temporary backup file: {local_backup_path}")
    except Exception as e:
        logging.warning(f"Failed to remove temporary backup file: {e}")


if __name__ == "__main__":
    main()