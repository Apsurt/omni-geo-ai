# Project Roadmap (Revised)

## Phase 1: Data Collection & Hemisphere Model
### Completed
- [x] Set up Google Street View API pipeline.
- [x] Design PostgreSQL database schema for panoramas.
- [x] Basic ViT-B/16 training template.

### Milestones
#### v1.0.0 - Data Pipeline MVP
**Target**: MM/YYYY  
**Features**:
- Cron job collecting 1,000 panoramas/day.
- Automated labeling (hemisphere, lat/lon).
- GCS + PostgreSQL storage system.

**Success Metrics**:
- 100,000 panoramas collected within 3 months.
- <5% data duplication rate.

#### v1.1.0 - Hemisphere Model
**Target**: MM/YYYY  
**Features**:
- ViT-B/16 model achieving ≥95% hemisphere accuracy.
- Model versioning with Weights & Biases.

**Success Metrics**:
- 95% test accuracy on held-out dataset.
- <2 sec inference time per image.

---

## Future Phases (Deferred)
### Phase 2: Country Recognition
- Train model to recognize 20 most common countries.
- Integrate reverse geocoding for labels.

### Phase 3: Advanced Masks
- Köppen climate classification.
- Urban/rural detection.

### Phase 4: Geoguessr Integration
- Private match automation via Selenium.
- Non-competitive gameplay mode.