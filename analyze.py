"""
Solution for Classification Task with XGBoost, Class Imbalance Handling, and Feature Interpretability

This script loads a dataset, preprocesses it, applies a classification model using XGBoost, and uses SHAP to explain model predictions and visualize feature importances.

Steps:
1. Load the training and test datasets.
2. Extract the target variable "Öbek İsmi" from the training data.
3. Encode the target variable using Label Encoding to convert string labels to numeric labels.
4. Drop unnecessary columns from both datasets.
5. Perform one-hot encoding for categorical variables.
6. Address class imbalance by oversampling the minority class using SMOTE.
7. Split the resampled training data into training and validation sets.
8. Perform hyperparameter tuning using GridSearchCV to find the best XGBoost model.
9. Train the best model on the entire resampled training dataset.
10. Make predictions on the test dataset.
11. Use SHAP to explain model predictions and visualize feature importances.
12. Decode the predicted labels back to their original string labels.
13. Save the predictions to a CSV file for submission.
14. Calculate the training accuracy of the final model.

This script ensures that the model is robust to class imbalance, and it provides insights into feature importances using SHAP values. It is suitable for a classification task with labeled data.

"""

import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from xgboost import XGBClassifier  # XGBoost classifier
from sklearn.metrics import accuracy_score  # For model evaluation
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import shap  # For feature interpretability
from sklearn.preprocessing import LabelEncoder  # For encoding target labels

# Load the dataset
train_df = pd.read_csv("data/train.csv")  # Load the training data
test_df = pd.read_csv("data/test_x.csv")  # Load the test data

# Extract the target variable "Öbek İsmi" from the training data
y_train = train_df["Öbek İsmi"]

# Encode the target variable using Label Encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Drop unnecessary columns (if needed)
X_train = train_df.drop(["Öbek İsmi", "index"], axis=1)  # Remove the target variable and index from the training data
X_test = test_df.drop(["index"], axis=1)  # Remove the index from the test data

# Perform one-hot encoding for categorical variables
X_train = pd.get_dummies(X_train, columns=["Cinsiyet", "Yaş Grubu", "Medeni Durum", "Eğitim Düzeyi", "İstihdam Durumu", "Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu", "Eğitime Devam Etme Durumu"])
X_test = pd.get_dummies(X_test, columns=["Cinsiyet", "Yaş Grubu", "Medeni Durum", "Eğitim Düzeyi", "İstihdam Durumu", "Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu", "Eğitime Devam Etme Durumu"])

# Address class imbalance using SMOTE
oversampler = SMOTE(sampling_strategy='minority')
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train_encoded)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [5, 7, 10],
    'n_estimators': [100, 200, 300]
}

xgb_classifier = XGBClassifier(random_state=42)
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

best_xgb_model = grid_search.best_estimator_

# Train the best model on the entire training dataset
best_xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test dataset
y_test_pred_encoded = best_xgb_model.predict(X_test)

# Decode the predicted labels
y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

# Save the predictions to a CSV file
test_df["Öbek İsmi"] = y_test_pred
submission_df = test_df[["index", "Öbek İsmi"]].rename(columns={'index': 'id'})
submission_df.to_csv("submission.csv", index=False)

# Calculate the training accuracy of the final model
train_accuracy = accuracy_score(y_train_resampled, best_xgb_model.predict(X_train_resampled))
print(f"Training Accuracy: {train_accuracy:.2f}")

# Use SHAP to explain model predictions and visualize feature importances
explainer = shap.Explainer(best_xgb_model)
shap_values = explainer.shap_values(X_val_split)
shap.summary_plot(shap_values, X_val_split)
