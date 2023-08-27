import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test_x.csv")

# Extract the target variable "Öbek İsmi" from the training data
y_train = train_df["Öbek İsmi"]

# Drop unnecessary columns (if needed)
X_train = train_df.drop(["Öbek İsmi", "index"], axis=1)
X_test = test_df.drop(["index"], axis=1)

# Perform one-hot encoding for categorical variables
X_train = pd.get_dummies(X_train, columns=["Cinsiyet", "Yaş Grubu", "Medeni Durum", "Eğitim Düzeyi", "İstihdam Durumu", "Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu", "Eğitime Devam Etme Durumu"])
X_test = pd.get_dummies(X_test, columns=["Cinsiyet", "Yaş Grubu", "Medeni Durum", "Eğitim Düzeyi", "İstihdam Durumu", "Yaşadığı Şehir", "En Çok İlgilendiği Ürün Grubu", "Eğitime Devam Etme Durumu"])

# Initialize a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Train the best model on the entire training dataset
best_rf_model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred = best_rf_model.predict(X_test)

# Save the predictions to a DataFrame
test_df["Öbek İsmi"] = y_pred

# Get only index and Öbek İsmi columns
test_df = test_df[["index", "Öbek İsmi"]]

# Rename the index column to id
test_df.rename(columns={'index': 'id'}, inplace=True)

# Save the test DataFrame with predictions to a CSV file
test_df.to_csv("submission.csv", index=False)

# Evaluate the model using accuracy and classification report
train_acc = accuracy_score(y_train, best_rf_model.predict(X_train))
print(f"Training Accuracy: {train_acc:.2f}")

# You can further evaluate the model as needed, e.g., using classification reports
# classification_rep = classification_report(y_true, y_pred)
# print(classification_rep)
