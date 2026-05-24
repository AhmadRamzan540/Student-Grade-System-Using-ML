import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ==============================================================================
# 1. SETUP AND INGESTION
# ==============================================================================
dataset_path = r"C:\Users\Microsoft\Downloads\StudentsPerformance.csv"
output_dir = "models"
plot_path = "accuracy_comparison.png"
passing_threshold = 50

print("="*80)
print(" STUDENTS GRADING SYSTEM - END-TO-END MACHINE LEARNING WORKFLOW")
print("="*80)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}. Check downloads folder.")

print(f"\n[STEP 1] Loading raw student dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)

# ==============================================================================
# 2. FEATURE ENGINEERING & TARGET LABEL DEFINITION
# ==============================================================================
print("[STEP 2] Performing target engineering (Math, Reading, & Writing >= 50)...")
df['passed'] = (
    (df['math score'] >= passing_threshold) & 
    (df['reading score'] >= passing_threshold) & 
    (df['writing score'] >= passing_threshold)
).astype(int)

# Set categorical and numerical feature lists
cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
num_cols = ['math score']

X = df[cat_cols + num_cols]
y = df['passed']

# Stratified 80/20 train-test split to keep class proportions balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" -> Training set size: {X_train.shape[0]} samples")
print(f" -> Testing set size:  {X_test.shape[0]} samples")
print(f" -> Target distribution (Passed vs Failed): {np.bincount(y)}")

# ==============================================================================
# 3. PREPROCESSING PIPELINE SPECIFICATION
# ==============================================================================
print("\n[STEP 3] Configuring ColumnTransformer for preprocessing...")
# Categorical columns: One-Hot Encoded, handles unseen categories gracefully
# Numerical columns: Standard Scaled (mean=0, std=1)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# ==============================================================================
# 4. MODEL SELECTION & HYPERPARAMETER GRID TUNING
# ==============================================================================
print("\n[STEP 4] Initiating model selection and cross-validated grid searches...")

models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [3, 5, 7, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7, None]
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [5, 10, 15, 20, 25]
        }
    },
    'Support Vector Classifier': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'classifier__C': [0.1, 0.2, 0.5, 1.0, 2.0],
            'classifier__kernel': ['linear', 'rbf']
        }
    }
}

results = {}

for name, config in models_config.items():
    print(f" -> Tuning hyperparameters for {name}...")
    
    # Construct unified scikit-learn Pipeline (Prevents Data Leakage)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])
    
    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        config['params'], 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    
    # Calculate performance metrics on the test set
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'pipeline': best_pipeline,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'best_params': grid_search.best_params_
    }
    
    print(f"    * Optimal params: {grid_search.best_params_}")
    print(f"    * Test Accuracy:  {acc:.4f} | F1-Score: {f1:.4f}")

# ==============================================================================
# 5. PERFORMANCE METRIC VISUALIZATION
# ==============================================================================
print("\n[STEP 5] Generating accuracy comparison chart...")
plot_data = []
for name, metrics in results.items():
    plot_data.append({
        'Model': name,
        'Accuracy': metrics['accuracy'] * 100
    })
df_plot = pd.DataFrame(plot_data).sort_values(by='Accuracy', ascending=False)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6), dpi=150)
colors = sns.color_palette("viridis", len(df_plot))

ax = sns.barplot(
    x='Accuracy', 
    y='Model', 
    data=df_plot, 
    palette=colors, 
    hue='Model', 
    legend=False
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', padding=5, fontsize=10, weight='bold', color='black')
    
plt.title("Student Performance Prediction - Accuracy Comparison", fontsize=14, weight='bold', pad=20)
plt.xlabel("Accuracy (%)", fontsize=11, labelpad=10)
plt.ylabel("Machine Learning Model", fontsize=11, labelpad=10)
plt.xlim(0, 105)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f" -> Accuracy comparison chart saved to: {plot_path}")

# ==============================================================================
# 6. MODEL COMPARISON SUMMARY
# ==============================================================================
summary_data = []
for name, metrics in results.items():
    summary_data.append({
        'Model Name': name,
        'Best Hyperparams': str(metrics['best_params']),
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1-Score': f"{metrics['f1_score']:.4f}"
    })
summary_df = pd.DataFrame(summary_data)

print("\n" + "="*95)
print(" MACHINE LEARNING PERFORMANCE SUMMARY")
print("="*95)
print(summary_df.to_string(index=False))
print("="*95)

# ==============================================================================
# 7. WINNING MODEL EVALUATION & SERIALIZATION
# ==============================================================================
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
winning_pipeline = results[best_model_name]['pipeline']
winning_accuracy = results[best_model_name]['accuracy']

print(f"\n[STEP 7] Winning Model Selected: {best_model_name} with Accuracy: {winning_accuracy:.4f}")
print("\nClassification Report for Winning Model:")

# Generate predictions again for printing report
y_pred_winning = winning_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_winning, target_names=['Fail', 'Pass']))

# Serialize best end-to-end model pipeline
os.makedirs(output_dir, exist_ok=True)
best_pipeline_path = os.path.join(output_dir, "best_pipeline.pkl")
with open(best_pipeline_path, 'wb') as f:
    pickle.dump(winning_pipeline, f)
print(f"[SAVED] Saved winning ML Pipeline to {best_pipeline_path}")

# Serialize all trained model pipelines
all_pipelines_path = os.path.join(output_dir, "all_pipelines.pkl")
with open(all_pipelines_path, 'wb') as f:
    pipelines_objs = {name: results[name]['pipeline'] for name in results}
    pickle.dump(pipelines_objs, f)
print(f"[SAVED] Saved all optimized pipelines to {all_pipelines_path}")

print("\nComplete ML pipeline workflow finished successfully!")
