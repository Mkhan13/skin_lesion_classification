# Pigmented Skin Lesion Classification

This repository contains code for classifying pigmented dermatological skin lesions using both a traditional machine learning (Random Forest) and a deep learning (Convolutional Neural Network) approach.

## Overview

Skin cancer is one of the most common forms of cancer in the world, and early detection of abnormal lesions is crucial for effective diagnosis and treatment. This project implements and compares two different machine learning approaches for the automated classification of pigmented skin lesion images into seven different diagnostic categories.

## Dataset

The dataset (dont forget to add my hyperlink here) used in this project consists of over 10,000 dermatoscopic images of pigmented skin lesions of the following types:
- **akiec**: Actinic keratoses and intraepithelial carcinoma
- **bcc**: Basal cell carcinoma
- **bkl**: Benign keratosis-like lesions
- **df**: Dermatofibroma
- **mel**: Melanoma
- **nv**: Melanocytic nevi
- **vasc**: Vascular lesions

The dataset is highly imbalanced, with the majority of images belonging to the 'nv' class.

## Models

### Random Forest with PCA

### Convolutional Neural Network (CNN)


## Optimization

## Results

### Overall Model Performance

| Model | Accuracy | Macro Avg F1-Score | Weighted Avg F1-Score |
|-------|----------|-------------------|----------------------|
| Random Forest | 0.6700 | 0.1172 | 0.5390 |
| CNN   | 0.7659 | 0.6486 | 0.7786 |

### Per-class Performance Metrics - Random Forest

| Class | Count in Dataset | Precision | Recall | F1-Score |
|-------|-----------------|-----------|--------|----------|
| akiec | 65 | 0.0000 | 0.0000 | 0.0000 |
| bcc   | 103 | 0.0000 | 0.0000 | 0.0000 |
| bkl   | 220 | 0.6667 | 0.0091 | 0.0179 |
| df    | 23 | 0.0000 | 0.0000 | 0.0000 |
| mel   | 223 | 0.0000 | 0.0000 | 0.0000 |
| nv    | 1341 | 0.6700 | 0.9993 | 0.8022 |
| vasc  | 28 | 0.0000 | 0.0000 | 0.0000 |

### Per-class Performance Metrics - CNN

| Class | Count in Dataset | Precision | Recall | F1-Score |
|-------|-----------------|-----------|--------|----------|
| akiec | 65 | 0.7200 | 0.5538 | 0.6261 |
| bcc   | 103 | 0.5984 | 0.7379 | 0.6609 |
| bkl   | 220 | 0.5240 | 0.6955 | 0.5977 |
| df    | 23 | 0.7368 | 0.6087 | 0.6667 |
| mel   | 223 | 0.4379 | 0.6009 | 0.5066 |
| nv    | 1341 | 0.9348 | 0.8233 | 0.8755 |
| vasc  | 28 | 0.6071 | 0.6071 | 0.6071 |

### Model Comparison

| Metric | CNN | Random Forest | Difference (in favor of) |
|--------|-----|---------------|--------------------------|
| Accuracy | 0.7659 | 0.6700 | 0.0959 (CNN) |
| Recall (macro) | 0.6610 | 0.1440 | 0.5170 (CNN) |
| F1 Score (macro) | 0.6486 | 0.1172 | 0.5315 (CNN) |

## Discussion

## Conclusion
