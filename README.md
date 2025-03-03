# Omni Geoguessr AI (Phase 1: Data & Hemisphere Model)

## Overview
This project focuses on building a pipeline to collect Google Street View panoramas and training a Vision Transformer (ViT) AI to predict the hemisphere (Northern/Southern) from images. This phase lays the groundwork for future geographic recognition models.

## Features
- **Automated Data Collection**: Fetches and labels panoramas using Google Street View API.
- **Hemisphere Prediction**: ViT-B/16 model trained to classify panoramas into Northern/Southern hemispheres.
- **Scalable Storage**: Images stored in Google Cloud Storage, metadata in PostgreSQL.

## Getting Started
### Prerequisites
- Python 3.10+
- PostgreSQL
- Google Cloud account with:
  - Street View Static API enabled
  - Geocoding API enabled
  - Google Cloud Storage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Apsurt/apsurt-omni-geo-ai.git
   cd apsurt-omni-geo-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Add your Google API key and PostgreSQL credentials to .env
   ```
4. Initialize the database:
   ```bash
   # Create PostgreSQL database
   createdb omni_geo_ai
   
   # Run the data collection script to initialize tables
   python scripts/fetch_panoramas.py --num_images 5 --dry_run
   ```

### Usage
#### Data Collection
Run the panorama scraper to collect a specific number of images:
```bash
python scripts/fetch_panoramas.py --num_images 1000
```

Advanced options:
```bash
# Control the ratio of Northern/Southern hemisphere images
python scripts/fetch_panoramas.py --num_images 100 --north_ratio 0.7

# Save collection statistics to a JSON file
python scripts/fetch_panoramas.py --num_images 50 --output_stats stats.json

# Enable verbose logging
python scripts/fetch_panoramas.py --num_images 20 --verbose
```

#### Database Backup
Backup the PostgreSQL database to Google Cloud Storage:
```bash
python scripts/backup_db.py
```

Backup options:
```bash
# Specify a different GCS bucket
python scripts/backup_db.py --bucket my-backup-bucket

# Custom backup directory path in GCS
python scripts/backup_db.py --backup_dir database/daily
```

#### Train Hemisphere Model
```bash
python train.py --model vit_hemisphere --epochs 50
```

## License
MIT License. See [LICENSE](LICENSE).

## Roadmap
See [ROADMAP.md](ROADMAP.md) for phase details.

## Contact
tymon.becella@gmail.com