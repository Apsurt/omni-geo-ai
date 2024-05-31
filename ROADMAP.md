# Project Roadmap

This document outlines the development roadmap for Omni Geoguessr AI project, detailing the phases, expected features, and timelines for achieving key milestones.

## Introduction

The project aims to develop a Vision Transformer AI capable of predicting geographic locations from streetview panoramas. The initial focus will be on country recognition, followed by integration with Geoguessr, enhanced mask predictions, and city recognition.

## [v1.0.0 - Basic Usable Product](https://github.com/Apsurt/omni-geo-ai/milestone/1)

**Objective**: Deliver a pre-trained and fine-tuned Vision Transformer AI capable of distinguishing between countries using streetview data from Google Maps API.

**Expected Features**:
- Fine-tuned visual transformer able to distinguish between countries.
- Automatically retrieve panoramas from Geoguessr.
- Predict the country visible in a photo.

**Tasks**:
- Use pre-trained Vision Transformer.
- Fine-tune model on streetview data.
- Implement panorama retrieval from Geoguessr.
- Develop country prediction mechanism.

**Metrics for Success**:
- Accuracy of country predictions.
- Performance benchmarks on test datasets.

**Achieved on**: [To be specified]

## [v2.0.0 - Seamless Integration in Geoguessr](https://github.com/Apsurt/omni-geo-ai/milestone/2)

**Objective**: Integrate the AI bot with Geoguessr for automatic gameplay.

**Expected Features**:
- User-friendly integration with Geoguessr.
- Fully automated gameplay: from panorama retrieval to guess submission.

**Tasks**:
- Develop integration API for Geoguessr.
- Create bot account management features.
- Implement automated game play logic.

**Metrics for Success**:
- Successful automation of private matches.
- User feedback and engagement levels.

**Achieved on**: [To be specified]

## [v3.0.0 - More Masks](https://github.com/Apsurt/omni-geo-ai/milestone/3)

**Objective**: Enhance the AI bot's performance with additional masks.

**Expected Features**:
- Recognition of Köppen climate classification.
- Height above sea level detection.
- Distance to the sea estimation.
- Urban-rural distinction.
- Combination of multiple masks for improved prediction accuracy.

**Tasks**:
- Fine-tune models for each new mask type.
- Implement parallel computation for mask prediction.
- Develop weighted sum algorithm for final mask combination.

**Metrics for Success**:
- Improvement in prediction accuracy.
- Efficiency of mask combination algorithm.

**Achieved on**: [To be specified]

## [v4.0.0 - City Recognition](https://github.com/Apsurt/omni-geo-ai/milestone/4)

**Objective**: Enhance the AI bot to recognize cities with populations over 100k inhabitants.

**Expected Features**:
- City recognition capabilities.
- Improved prediction accuracy for urban areas.

**Tasks**:
- Develop and fine-tune city recognition model.
- Integrate city recognition with existing prediction framework.
- Validate performance with real-world data.

**Metrics for Success**:
- Accuracy of city recognition.
- Impact on overall prediction performance.

**Achieved on**: [To be specified]

## Dependencies and Risks

- **Data availability**: Reliance on Google Maps API and Geoguessr data.
- **Technical challenges**: Model accuracy and computational efficiency.