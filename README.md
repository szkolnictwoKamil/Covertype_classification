# Forest Cover Type Classification

## Overview
This project focuses on classifying forest cover types using a dataset containing various environmental features. The classification is performed using multiple machine learning approaches, including heuristic methods, decision trees, k-nearest neighbors (k-NN), random forests, and neural networks.

## Dataset
The dataset used in this project is `covtype.data`, which contains multiple numerical and categorical features that describe different forest cover types. The dataset is preprocessed by handling class imbalances and scaling numerical attributes.

## Project Structure
The project consists of two main scripts:

### 1. `data_analysis.py`
This script performs an initial exploratory data analysis to understand the dataset. The key steps include:
- Correlation matrix visualization to detect relationships between features.
- Distribution analysis of cover types to check for class imbalances.
- Identification of outliers, missing values, and duplicate records.

### 2. `classification_methods.py`
This script implements various classification models to predict forest cover types:

#### **Heuristic Method**
- Computes the mean values for each feature per class and predicts the class based on the minimal difference in means.

#### **Random Forest Classifier**
- Uses a balanced Random Forest model with 100 estimators to classify forest cover types.

#### **k-Nearest Neighbors (k-NN) Classifier**
- Implements k-NN with 3 neighbors for classification.

#### **Neural Network Model**
- A feedforward neural network with multiple dense layers is trained using the Adam optimizer.
- The best hyperparameters (batch size and epochs) are determined using GridSearchCV.

## Model Evaluation
Each model is evaluated using:
- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
