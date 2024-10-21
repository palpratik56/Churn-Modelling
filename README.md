## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Model Development](#model-development)
6. [Installation and Setup](#installation-and-setup)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)

## Project Overview

This project provides a **straightforward yet effective implementation** of a **machine learning model** to predict **customer churn**. Customer churn refers to when customers stop doing business with a company, and predicting churn can help businesses take proactive measures to retain customers. This model is designed to analyze customer data and predict which customers are likely to churn based on various features such as age, geography, balance, etc.

The model was built using various machine learning algorithms and achieved high accuracy in predicting churn.

## Features

- **Customer churn prediction** based on input features like geography, age, and balance.
- Utilizes a range of machine learning algorithms including **Logistic Regression**, **Random Forest**, and **Artificial Neural Networks (ANN)**.
- **Data preprocessing** steps like feature scaling and encoding categorical variables.
- **Exploratory Data Analysis (EDA)** to understand feature relationships and distributions.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning Libraries:** Scikit-learn, TensorFlow/Keras (for ANN), XGBoost
- **Data Analysis Libraries:** Pandas, NumPy
- **Data Visualization Libraries:** Matplotlib, Seaborn
- **IDE:** Jupyter Notebook

## Dataset

The dataset used in this project includes customer information such as:

- **Geography**
- **Customer Age**
- **Balance**
- **Credit Score**
- **Is Active Member**
- **Exited (Target variable indicating churn)**

The dataset can be found [here](https://www.kaggle.com/shubhendra247/customer-churn-prediction).

## Model Development

The following machine learning algorithms were used for customer churn prediction:

1. **Logistic Regression**: A simple yet effective classification algorithm.
2. **Random Forest Classifier**: An ensemble method that uses multiple decision trees to improve prediction accuracy.
3. **Artificial Neural Network (ANN)**: A deep learning model designed to capture complex patterns in the data.

### Key Steps:
- **Data Preprocessing**: Handle missing values, encode categorical variables, and scale the features.
- **Exploratory Data Analysis (EDA)**: Analyze distributions, correlations, and feature importance.
- **Model Training**: Train multiple machine learning models and evaluate their performance using metrics like accuracy, precision, and recall.
- **Model Evaluation**: Cross-validation and confusion matrix analysis.

## Installation and Setup

To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/palpratik56/Churn-Modelling.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Churn-Modelling
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Jupyter Notebook:**
    Open the notebook (`Churn_Modelling.ipynb`) in Jupyter to run the models and view results.

## Usage

Once the environment is set up, you can run the notebook to train the model and predict customer churn. The notebook includes:

1. **Data Preprocessing:** Feature scaling, encoding categorical variables.
2. **Model Training:** Train Logistic Regression, Random Forest, and ANN models.
3. **Model Evaluation:** Evaluate the trained models on test data and visualize performance metrics.

## Results

The following table summarizes the performance of different models used in the project:

| Model                 | Accuracy | Precision | Recall |
|-----------------------|----------|-----------|--------|
| Logistic Regression    | 84%      | 0.79      | 0.85   |
| Random Forest          | 88%      | 0.81      | 0.89   |
| Artificial Neural Network | 90%   | 0.83      | 0.92   |

### Visualizing Feature Importance:

A key part of the project involves analyzing the importance of each feature in predicting churn. The feature importance plot for Random Forest is displayed in the notebook.

## Contributing

Contributions are welcome! If you want to contribute to this project, you can:

1. Fork the project.
2. Create a feature branch (`git checkout -b feature/newFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/newFeature`).
5. Open a pull request.
