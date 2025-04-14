# Pigmented Skin Lesion Classification

This repository contains code for classifying pigmented dermatological skin lesions using both a traditional machine learning (Random Forest) and a deep learning (Convolutional Neural Network) approach.

## Overview

Skin cancer is one of the most common forms of cancer in the world, and early detection of abnormal lesions is crucial for effective diagnosis and treatment. This project implements and compares two different machine learning approaches for the automated classification of pigmented skin lesion images into seven different diagnostic categories.

## Dataset

The [dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
 used in this project is the HAM10000 datasete which consists of over 10,000 dermatoscopic images of pigmented skin lesions of the following types:

| Class | Description | Count |
|-------|-------------|-------|
| **nv** | Melanocytic nevi | 6705 |
| **mel** | Melanoma | 1113 |
| **bkl** | Benign keratosis-like lesions | 1099 |
| **bcc** | Basal cell carcinoma | 514 |
| **akiec** | Actinic keratoses and intraepithelial carcinoma | 327 |
| **vasc** | Vascular lesions | 142 |
| **df** | Dermatofibroma | 115 |

The dataset is highly imbalanced, with the majority of images belonging to the 'nv' class.

## Models

### Random Forest with PCA
The Random Forest model uses a traditional machine learning approach:

1. Images are flattened into 1D vectors
2. Principal Component Analysis (PCA) reduces dimensionality to 200 components
3. A Random Forest classifier with 300 trees is trained on the reduced features
4. Class weights are balanced to address the dataset imbalance

The Random Forest implementation provides a baseline model that requires less computational resources compared to deep learning approaches.

### Convolutional Neural Network (CNN)
The CNN model follows a standard architecture for image classification:

1. Three convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
2. Fully connected layers with dropout for regularization
3. Batch normalization to stabilize training
4. He initialization for better gradient flow

The network gradually increases feature map depth (64→128→256) while reducing spatial dimensions through pooling operations.

## Optimization

**Random Forest:**

* PCA for feature reduction
* Class-balanced training to handle imbalance

**CNN:**

* Dropout and batch normalization for regularization
* Class-weighted loss to address class imbalance
* Gradient clipping and dynamic learning rate adjustment based on validation loss plateau

## Results

### MOdel Performance Summary

| Model | Accuracy | Macro Avg F1-Score | Weighted Avg F1-Score |
|-------|----------|-------------------|----------------------|
| Random Forest | 0.6700 | 0.1172 | 0.5390 |
| CNN   | 0.7659 | 0.6486 | 0.7786 |

### Per-class Performance Metrics - Random Forest

| Class | Count in Test Set | Precision | Recall | F1-Score |
|-------|-----------------|-----------|--------|----------|
| akiec | 65 | 0.0000 | 0.0000 | 0.0000 |
| bcc   | 103 | 0.0000 | 0.0000 | 0.0000 |
| bkl   | 220 | 0.6667 | 0.0091 | 0.0179 |
| df    | 23 | 0.0000 | 0.0000 | 0.0000 |
| mel   | 223 | 0.0000 | 0.0000 | 0.0000 |
| nv    | 1341 | 0.6700 | 0.9993 | 0.8022 |
| vasc  | 28 | 0.0000 | 0.0000 | 0.0000 |

### Per-class Performance Metrics - CNN

| Class | Count in Test Set | Precision | Recall | F1-Score |
|-------|-----------------|-----------|--------|----------|
| akiec | 65 | 0.7200 | 0.5538 | 0.6261 |
| bcc   | 103 | 0.5984 | 0.7379 | 0.6609 |
| bkl   | 220 | 0.5240 | 0.6955 | 0.5977 |
| df    | 23 | 0.7368 | 0.6087 | 0.6667 |
| mel   | 223 | 0.4379 | 0.6009 | 0.5066 |
| nv    | 1341 | 0.9348 | 0.8233 | 0.8755 |
| vasc  | 28 | 0.6071 | 0.6071 | 0.6071 |

### Model Comparison

| Metric | Random Forest | CNN | Difference (in favor of) |
|--------|-----|---------------|--------------------------|
| Accuracy | 0.6700 | 0.7659 | 0.0959 (CNN) |
| Recall | 0.1440 | 0.6610 | 0.5170 (CNN) |
| F1 Score | 0.1172 | 0.6486 | 0.5315 (CNN) |

## Discussion
CNN significantly outperforms Random Forest across all metrics. It achieves 76.6% accuracy vs. 67.0%, and a macro F1-score of 0.6486 vs. 0.1172, which highlights its ability to generalize across all classes, not just the majority. Random Forest performs well only on the 'nv' class (F1: 0.8022), which is the majority type in the dataset. It fails entirely (F1: 0.0000) on all other classes, which relects high bias toward the majority class. Random Forest struggles with minority classes due to its reliance on reduced linear features from PCA. CNN’s worst F1 is on 'mel' (0.5066), which is a difficult to identify because of visual similarity with 'nv'.

## Conclusion
CNN is the better model for classification of pigmented skin lesions besed on the results from this dataset. It has a 53% gain in F1 and a 9.6% gain in accuracy compared to non deep learning models. IN addition, Random Forest is not suitable for imbalanced image classification tasks without additional feature engineering or resampling techniques because it heavily relies on having balanced features.
