import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, is_train=True):
    if is_train:
        y = df["Öbek İsmi"]
        df = df.drop(["Öbek İsmi", "index"], axis=1)
    else:
        df = df.drop(["index"], axis=1)

    df = pd.get_dummies(
        df,
        columns=[
            "Cinsiyet",
            "Yaş Grubu",
            "Medeni Durum",
            "Eğitim Düzeyi",
            "İstihdam Durumu",
            "Yaşadığı Şehir",
            "En Çok İlgilendiği Ürün Grubu",
            "Eğitime Devam Etme Durumu",
        ],
    )

    if is_train:
        return df, y
    else:
        return df


def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
    }

    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def save_predictions(test_df, y_pred):
    test_df["Öbek İsmi"] = y_pred
    submission_df = test_df[["index", "Öbek İsmi"]].rename(columns={"index": "id"})
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    train_file_path = "data/train.csv"
    test_file_path = "data/test_x.csv"

    train_df = load_data(train_file_path)
    test_df = load_data(test_file_path)

    X_train, y_train = preprocess_data(train_df, is_train=True)
    X_test = preprocess_data(test_df, is_train=False)

    best_rf_model = train_model(X_train, y_train)

    y_pred = best_rf_model.predict(X_test)

    save_predictions(test_df, y_pred)

    train_acc = accuracy_score(y_train, best_rf_model.predict(X_train))
    print(f"Training Accuracy: {train_acc:.2f}")
