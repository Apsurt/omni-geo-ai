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
- Google Cloud account (for Street View API)

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

### Usage
#### Data Collection
Run the daily panorama scraper:
```bash
python scripts/fetch_panoramas.py --num_images 1000
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