# SMS Spam Detection Using Machine Learning Models

## Overview

This project focuses on detecting spam SMS messages using four machine learning algorithms: **K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Random Forest**, and **XGBoost**. The models are trained and evaluated on a publicly available SMS spam dataset. The goal is to identify the most accurate and robust model for spam detection.

Data Retrieval Course Dr. Mohammadreza Shams
## Features

- **Data Preprocessing**: Text cleaning, TF-IDF vectorization, and label encoding.
- **Model Training**: Implementation of KNN, Logistic Regression, Random Forest, and XGBoost.
- **Hyperparameter Tuning**: GridSearchCV for finding optimal parameters.
- **Cross-Validation**: 10-fold stratified cross-validation for robust evaluation.
- **Performance Metrics**: Accuracy, standard deviation of accuracy across folds.

## Installation

To run this project, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
```

## Usage

1. **Data Loading**: Load the SMS spam dataset from the provided URL.
2. **Preprocessing**: Clean the text data and convert it into numerical features using TF-IDF.
3. **Model Training**: Train and tune the models using GridSearchCV.
4. **Evaluation**: Perform 10-fold cross-validation and compare model performances.
5. **Visualization**: Plot the mean accuracy of each model for comparison.

## Results

The following table summarizes the performance of the models:

| Model                  | Mean Accuracy | Std Accuracy |
|----------------------|---------------|--------------|
| KNN                   | 0.9587        | 0.0060       |
| Logistic Regression   | 0.9846        | 0.0043       |
| Random Forest         | 0.9826        | 0.0060       |
| XGBoost               | 0.9747        | 0.0055       |

**Analysis**:  
Logistic Regression outperforms other models with the highest mean accuracy (0.9846) and the lowest standard deviation (0.0043). Random Forest follows closely with a mean accuracy of 0.9826. XGBoost and KNN also perform well but are slightly less accurate.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additional features.

## Acknowledgments

- **Dataset**: SMS Spam Collection Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Libraries**: Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib

## Contact

For any inquiries, please contact **Iman Khalilorahmani** at [imankhtech@gmail.com](mailto:imankhtech@gmail.com).

---
