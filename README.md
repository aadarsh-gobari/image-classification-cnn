# Image Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

## Dataset
- CIFAR-10 (TensorFlow built-in)
- 60,000 color images
- 10 classes

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Current Status
- Dataset loaded
- Images normalized
- Sample visualizations created

## Model Architecture
- Three convolutional layers with ReLU activation
- Max pooling for spatial reduction
- Fully connected layers with dropout for regularization
- Softmax output for multi-class classification

## Model Training
- CNN trained using Adam optimizer
- Early stopping used to reduce overfitting
- Training and validation curves generated

## Model Evaluation
- Evaluated model on unseen test data
- Visualized correct and incorrect predictions
- Identified class-level challenges in image classification
