# End-to-End Machine Learning Pipeline: Regression & Classification Model Comparison

## Project Overview
This project demonstrates a complete machine learning workflow using Python and scikit-learn. It covers **regression** and **classification** tasks, evaluates multiple algorithms, compares their performance, and explores ensemble learning methods.

The goal is to understand **which models perform best for different problems** and how preprocessing, feature scaling, and hyperparameters impact results.

---

## Datasets Used

### 1. California Housing Dataset (Regression)
- Predicts median house values in California districts
- Features include: median income, house age, number of rooms, population, and location-related data

### 2. Breast Cancer Dataset (Classification)
- Predicts tumor type (malignant or benign)
- Features include medical measurements of breast tissue

---

## Libraries & Tools
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook

---

## Models Implemented

### Regression Models:
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

### Classification Models:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Gradient Boosting
- Voting Ensemble Classifier

---

## Workflow

1. **Data Loading & Exploration**
   - Understand dataset structure, features, target variable
   - Check for missing values and summary statistics

2. **Data Preprocessing**
   - Handling missing values
   - Feature scaling using StandardScaler
   - Train-test split

3. **Model Training & Evaluation**
   - Regression: MSE, RMSE, R² Score
   - Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   - Diagnostic plots (residuals, actual vs predicted)

4. **Hyperparameter Experimentation**
   - KNN with multiple K values
   - SVM with linear and RBF kernels
   - Tree-based model depths

5. **Ensemble Learning**
   - Random Forest (Bagging)
   - Gradient Boosting
   - Voting Classifier

6. **Visualization & Analysis**
   - Feature correlation heatmap
   - Confusion matrix heatmap
   - Accuracy comparison charts
   - KNN accuracy vs K value

---

## Key Insights

- Tree-based models like **Random Forest** generally outperform single Decision Trees due to reduced overfitting.
- **SVM and KNN** are sensitive to feature scaling.
- Boosting algorithms can outperform bagging by sequentially correcting errors.
- Ensemble models (Voting) can improve performance by combining strengths of multiple classifiers.
- Logistic Regression is fast and suitable for **real-time predictions**.
- Correlated features may reduce performance for Naive Bayes and Linear models.
