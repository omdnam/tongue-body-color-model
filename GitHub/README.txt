# Tongue Body Color Discrimination Model for Chronic Fatigue Patients

This repository contains the Python scripts used for the data processing, feature extraction, and machine learning model training as described in the manuscript, "Development of models for discriminating tongue body colors among chronic fatigue patients using machine learning."

## Scripts Execution Order

The scripts are numbered to indicate the order of execution:

1.  `01_calculate_reliability.py`: Performs the intra-rater (Weighted Kappa) and inter-rater (Fleiss' Kappa) reliability analyses for the clinician assessments, as detailed in the Methods section.
2.  `02_generate_train_test_split.py`: Documents the initial stratified random split of the 198 original images into a training set (158) and a test set (40). This ensures reproducibility of the exact dataset split used for the analysis.
3.  `03_dataset_preparation.py`: Organizes the raw tongue image dataset into a structured format based on the pre-defined train/test split.
4.  `04_image_augmentation.py`: Applies various augmentation techniques to the training images to enhance model robustness.
5.  `05_feature_extraction.py`: Extracts key colorimetric features (e.g., CIE-Lab values) from all images and saves them to a CSV file.
6.  `06_decision_tree_model.py`: Trains and evaluates the Decision Tree classifier.
7.  `07_random_forest_model.py`: Trains and evaluates the Random Forest classifier.
8.  `08_xgboost_model.py`: Trains and evaluates the XGBoost classifier.
