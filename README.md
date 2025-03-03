Iris Dataset Classification Project

Overview

This project is a classification task using the Iris dataset, where we predict the species of an Iris flower based on its sepal length, sepal width, petal length, and petal width. The dataset is preprocessed, missing values are handled, and a machine learning model is trained for classification.

Dataset

The Iris dataset consists of 150 samples with 4 numerical features and 1 categorical target variable (species).

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target Classes:

Setosa (0)

Versicolor (1)

Virginica (2)

Some missing values are handled using the mean imputation method.

Project Steps

Load the dataset from CSV.

Handle missing values using the mean of each column.

Encode categorical labels using LabelEncoder().

Normalize features using StandardScaler().

Split data into training (80%) and testing (20%) sets.

Train a classification model (Logistic Regression, Random Forest, etc.).

Evaluate the model using accuracy, confusion matrix, and classification report.

Visualize results (pair plots, confusion matrix heatmap, feature importance).

Requirements

Install dependencies using:

pip install pandas numpy scikit-learn seaborn matplotlib

Usage

Run the Python script to train and evaluate the model:

python Iris.py

Model Evaluation

Accuracy is measured using accuracy_score().

Confusion Matrix is used to analyze misclassifications.

Feature Importance is visualized (for tree-based models).

Decision Boundary Plot is used (after PCA reduction to 2D).

Visualizations

This project includes the following visualizations:

Pairplot to visualize relationships between features.

Confusion Matrix Heatmap to evaluate classification performance.




