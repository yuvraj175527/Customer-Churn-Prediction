from src.business_impact import calculate_business_impact
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.feature_engineering import create_features


# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("dataset/churn.csv")

df.drop(columns=["customerID"], errors="ignore", inplace=True)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Convert target into numerical values
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.dropna(subset=["Churn"], inplace=True)


# 2. Feature engineering
print("Creating features...")

df = create_features(df)


# 3. Split dataset
X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Preprocessing
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Define machine learning models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"model__C": [0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "model__n_estimators": [100, 150],
            "model__max_depth": [10, None]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1]
        }
    )
}


# 6. Train and tune models
best_score = -1
best_model = None
results = []

print("\nStarting model training...\n")

for name, (model, parameters) in models.items():

    print("Training:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    trained_model = grid.best_estimator_

    predictions = trained_model.predict(X_test)
    probabilities = trained_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    print("\n", name)
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    if auc > best_score:
        best_score = auc
        best_model = trained_model


# 7. Save best model
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully!")


# 8. Model comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# 9. Final evaluation
final_predictions = best_model.predict(X_test)

print("\nBEST MODEL CONFUSION MATRIX")
print(confusion_matrix(y_test, final_predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    final_predictions,
    zero_division=0
))

print("\nBest ROC-AUC:", round(best_score, 4))
print("Training completed successfully!")
