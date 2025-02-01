# Supervised Machine Learning  

This repository contains a Jupyter Notebook demonstrating various supervised machine learning techniques using multiple classification algorithms.

## Overview  

The notebook focuses on supervised learning using different classifiers, applying feature scaling and categorical encoding, tuning hyperparameters, and evaluating models using performance metrics. The dataset used in this project is `spam.dat`, which is preprocessed and analyzed to build predictive models.  

## Technologies Used  

The following technologies and libraries were used in this project:  

- **CatBoost** - A gradient boosting library optimized for categorical data.  
- **Scikit-Learn** - Used for dataset preprocessing, splitting, and evaluation.  
- **Pandas & NumPy** - For data manipulation and numerical operations.  

## Machine Learning Methods  

The following machine learning techniques are covered:  

- **Supervised Learning** - Training models on labeled data to make predictions.  
- **Train/Test Splitting** - Dividing the dataset into training and testing sets for evaluation.  
- **Feature Scaling** - Standardizing numerical data using normalization or standardization.  
- **Categorical Encoding** - Transforming categorical features using label encoding and one-hot encoding.  
- **Hyperparameter Tuning** - Optimizing model parameters using GridSearch and other techniques.  
- **Model Evaluation Metrics** - Using accuracy, precision, recall, F1-score, and confusion matrix to assess model performance.  

## Classifiers Implemented  

Several classification models were trained and compared:  

- **CatBoostClassifier** - A powerful gradient boosting model designed for categorical features.  
- **XGBoost Classifier** - An optimized distributed gradient boosting model.  
- **Random Forest Classifier** - A popular ensemble learning technique using multiple decision trees.  
- **Logistic Regression** - A simple but effective linear classification algorithm.  
- **K-Nearest Neighbors (KNN)** - A distance-based algorithm used for classification tasks.  
- **Gaussian Naive Bayes** - A probabilistic classifier based on Bayes’ theorem.  

## Installation  

To run the notebook, install the required dependencies:  

```bash
pip install catboost scikit-learn pandas numpy matplotlib
