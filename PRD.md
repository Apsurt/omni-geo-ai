### **Product Requirements Document (PRD): Omni Geoguessr AI (Phase 1)**  
**Version**: Data Collection & Hemisphere Model  
**Date**: 03.03.2025

---

## **1. Objective & Scope**  
**Objective**: Build a scalable pipeline to collect and store labeled Google Street View panoramas, and train a Vision Transformer (ViT) model to predict hemispheres (Northern/Southern) from images.  

**Scope**:  
- **Data Collection**: Automatically gather and label panoramas globally via Google Street View API.  
- **Storage**: Organize data in a structured database for easy retrieval.  
- **Hemisphere Model**: Train a ViT-B/16 model to classify hemispheres with ≥95% accuracy.  

**Out of Scope**:  
- Country/city recognition.  
- Geoguessr bot automation.  

---

## **2. Data Collection & Storage**  
### **Data Pipeline**  
1. **Panorama Acquisition**:  
   - **Tool**: Google Street View Static API ([documentation](https://developers.google.com/maps/documentation/streetview)).  
   - **Parameters**:  
     - Random `lat/lon` coordinates within Google Street View coverage.  
     - Image size: `640x640` (max resolution without premium tier).  
     - Format: JPEG.  
   - **Quota Management**:  
     - Max **28,000 requests/month** (within $200 free tier).  
     - Prioritize regions with high geographic diversity.  

2. **Labeling**:  
   - **Automated Labels**:  
     - **Hemisphere**: Northern (`lat ≥ 0`) or Southern (`lat < 0`).  
     - **Metadata**: Latitude, longitude, country (via reverse geocoding).  
   - **Tools**:  
     - Reverse geocoding: Google Maps Geocoding API.  
     - Climate/city labels: Deferred for future phases.  

3. **Database Schema**:  
   - **Database**: PostgreSQL (structured, scalable).  
   - **Tables**:  
     ```sql  
     CREATE TABLE panoramas (  
         id SERIAL PRIMARY KEY,  
         image_path VARCHAR(255),  -- e.g., "gs://bucket/hemisphere/north/IMG_001.jpg"  
         latitude FLOAT,  
         longitude FLOAT,  
         hemisphere VARCHAR(10),  -- "north" or "south"  
         country_code VARCHAR(2),  
         date_added TIMESTAMP  
     );  
     ```  
   - **Storage**:  
     - Images: Google Cloud Storage (GCS) bucket, organized by hemisphere.  
     - Metadata: PostgreSQL database (hosted locally or on GCP).  

4. **Data Quality**:  
   - **Filters**:  
     - Exclude coordinates with no Street View coverage.  
     - Remove duplicate panoramas (hash-based deduplication).  
   - **Batch Processing**:  
     - Daily cron job to collect 1,000 panoramas (avoid quota exhaustion).  

---

## **3. Hemisphere Prediction Model**  
### **Model Architecture**  
- **Base Model**: `ViT-B/16` (pre-trained on ImageNet-21k).  
- **Modifications**:  
  - Replace final classification layer with **2-node output** (Northern/Southern).  
  - Freeze all layers except the final transformer block and head.  

### **Training Process**  
1. **Dataset Splits**:  
   - Train: 70% | Validation: 15% | Test: 15% (stratified by hemisphere).  
2. **Augmentation**:  
   - Random cropping, rotation (±15°), brightness/contrast adjustment.  
3. **Hyperparameters**:  
   - Optimizer: AdamW (`lr=3e-5`).  
   - Batch size: 32 (adjust for Apple Metal memory limits).  
   - Epochs: 50 (early stopping if validation loss plateaus).  
4. **Hardware**:  
   - Apple Metal (M1/M2 GPU) via PyTorch `MPS` backend.  

### **Evaluation**  
- **Primary Metric**: Accuracy (≥95% on test set).  
- **Confusion Matrix**: Track false north/south predictions.  
- **Latency**: <2 sec per image (local inference).  

### **Model Storage**  
- **Formats**:  
  - PyTorch `.pt` (training checkpoints).  
  - ONNX (optional, for compatibility).  
- **Versioning**:  
  - Weights & Biases (W&B) for tracking experiments.  

---

## **4. Future-Proofing**  
### **Data Collection for Advanced Models**  
- **Additional Metadata**: Collect but defer processing:  
  - Elevation (via Google Elevation API).  
  - Köppen climate zone (from latitude/longitude lookup tables).  
- **Database Backups**:  
  - Nightly exports to GCS (`.sql` dumps + TFRecords).  

### **Scalability**  
- **Incremental Updates**:  
  - Database supports appending new panoramas without reprocessing old data.  
- **Pipeline Flexibility**:  
  - Modular codebase to add future labeling (e.g., country recognition).  

---

## **5. Risks & Mitigations**  
| **Risk** | **Mitigation** |  
|----------|----------------|  
| Google API quota exhaustion | Distribute requests evenly across days/regions. |  
| Biased hemisphere distribution (e.g., more Northern samples) | Stratified sampling during data collection. |  
| Apple Metal GPU limitations (small batch sizes) | Gradient accumulation for stable training. |  
| Database corruption | Daily backups to GCS. |  

---

## **6. Milestones**  
| **Milestone** | **Deliverables** |  
|----------------|-------------------|  
| Data Pipeline MVP | Functional cron job collecting 1,000 panoramas/day. |  
| Hemisphere Model v1 | ViT-B/16 model with ≥95% accuracy. |  
| Database Optimization | Indexed queries, backup system. |  

---

## **7. Tools & Technologies**  
- **Data Pipeline**: Python, Google Maps APIs, PostgreSQL, SQLAlchemy.  
- **Training**: PyTorch, Apple Metal, W&B.  
- **Storage**: Google Cloud Storage, PostgreSQL.