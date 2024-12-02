# Diabetes Prediction using SVM
- Machine learning model using Support Vector Machine (SVM) to predict diabetes based on diagnostic measurements.
  
## Overview
- This project implements a binary classification model to predict whether a patient has diabetes using clinical measurements. The model achieves training accuracy of ~78% using a linear SVM classifier.

## Dataset
### The model uses the Pima Indians Diabetes Database with the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target variable: 1 for diabetes, 0 for no diabetes)

## Requirements
```python
numpy
pandas
scikit-learn
```
## Model Details
- Algorithm: Support Vector Machine (SVM) with linear kernel
- Data split: 80% training, 20% testing
- Features are standardized using StandardScaler
- Stratified sampling used to maintain class distribution

## Required libraries
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
```
## Performance
- Training Accuracy: ~78%
- Testing Accuracy: ~77%

## Future Improvements
- Implement cross-validation
- Try different kernel functions
- Feature selection/engineering
- Hyperparameter tuning
