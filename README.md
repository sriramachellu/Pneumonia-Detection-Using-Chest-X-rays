# Pneumonia Detection Using Chest X-Rays

This repository presents a deep learning project for the automatic detection of pneumonia from pediatric chest X-ray images using Convolutional Neural Networks (CNNs). The project was conducted as part of the graduate course *Data Science Meets Health Science* at Florida State University.

## Project Overview

Pneumonia is a serious respiratory illness, especially in children under five. Accurate and early detection is crucial for timely treatment and preventing complications. This project leverages computer vision and deep learning techniques to classify chest X-ray images into two categories: Normal and Pneumonia.

## Objective

- Automate the identification of pneumonia using chest X-rays.
- Reduce diagnostic time and support clinical decision-making.
- Evaluate performance through standard classification metrics.
- Explore generalization capabilities and model limitations.

## Dataset

- **Source**: Guangzhou Women and Children’s Medical Center via Kaggle  
  [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,863
- **Split**:
  - Training: 5,160 images
  - Validation: 72 images
  - Testing: 624 images
- **Classes**: Normal, Pneumonia
- **Image Format**: JPEG, grayscale, 224x224 pixels

## Methodology

### Data Preprocessing

The notebook includes a complete data preprocessing pipeline:

- Resizing all images to 224x224 pixels
- Grayscale conversion
- Normalization with mean = 0.5, std = 0.5
- Image augmentation using `RandomHorizontalFlip`
- Batch loading using a custom PyTorch `Dataset` class

### Model Architecture

The model is a custom CNN implemented using PyTorch:

- 3 convolutional layers with ReLU activation and MaxPooling
- Flattened output passed through:
  - FC1: Fully connected layer with 128 neurons
  - FC2: Output layer with sigmoid activation
- Optimizer: Adam
- Loss Function: Binary Cross Entropy (BCELoss)
- Input Channels: 1 (Grayscale)
- Epochs: 5
- Batch Size: 32

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The `project.ipynb` notebook includes plots for training and validation loss, accuracy, and other performance metrics.

## Results

| Metric       | Training     | Validation   | Test        |
|--------------|--------------|--------------|-------------|
| Accuracy     | 97.83%       | 98.61%       | 79.17%      |
| Precision    | ~98%         | ~97%         | Moderate    |
| Recall       | ~98%         | ~96%         | Moderate    |
| F1-Score     | ~97%         | ~96.5%       | Moderate    |

- Most classification errors occurred as false positives.
- The validation accuracy peaked in the fourth epoch.
- Slight overfitting observed, indicating scope for improved generalization.

## Discussion

The CNN model demonstrated effective classification of pediatric X-rays. However, a few limitations were noted:

- Slight overfitting due to dataset imbalance and low validation volume
- Limited generalization to unseen test data
- Black-box nature of CNNs makes clinical interpretability difficult

Comparison with baseline models (Logistic Regression, SVM, Random Forest) shows that CNNs offer superior performance in image classification but require more data and model transparency.

## Future Work

- Apply transfer learning using ResNet or DenseNet
- Integrate explainability tools like Grad-CAM
- Use larger, diverse datasets for better generalization
- Develop web-based or mobile deployment pipelines
- Explore hybrid models combining CNNs with traditional ML methods


## Authors

- Srirama Murthy Chellu (SC23BK)
- Sameera Rompicherla (SR23BA)
- Yashwanth Gowram (YG23G)

Instructor: Dr. Olmo Zavala Romero  
Course: Data Science Meets Health Science  
Institution: Florida State University  
Date: May 2024

## References

1. Rajpurkar et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-rays"
2. Wang et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database"
3. Stephen et al. (2019). "An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare"

---

This project illustrates the potential of deep learning in medical diagnostics and provides a foundation for scalable, automated disease detection systems in clinical settings.

