# 1. IMPORT LIBRARIES

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report


# 2. LOAD DATA


dataset_path = r"C:\Users\Microsoft\Downloads\StudentsPerformance.csv"
df = pd.read_csv(dataset_path)

print("Data Loaded:", df.shape)


# 3. DATA PREPROCESSING (TARGET + FEATURES)


df["passed"] = (
    (df["math score"] >= 50) &
    (df["reading score"] >= 50) &
    (df["writing score"] >= 50)
).astype(int)

categorical_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

numerical_features = ["math score"]

X = df[categorical_features + numerical_features]
y = df["passed"]


# 4. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# 5. FEATURE SCALING & ENCODING


preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("numerical", StandardScaler(), numerical_features)
    ]
)

# 6. MODEL TRAINING


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():

    print("\nTraining Model:", name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": pipeline,
        "accuracy": acc
    }

    print("Accuracy:", acc)


# 7. EVALUATION


best_model_name = max(results, key=lambda x: results[x]["accuracy"])

print("\nBest Model:", best_model_name)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        results[best_model_name]["model"].predict(X_test),
        target_names=["Fail", "Pass"]
    )
)


# 8. SAVE MODEL


os.makedirs("models", exist_ok=True)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(results[best_model_name]["model"], f)

with open("models/best_pipeline.pkl", "wb") as f:
    pickle.dump(results[best_model_name]["model"], f)

print("\nModel saved successfully as models/best_model.pkl and models/best_pipeline.pkl")