import os
import pickle
import pandas as pd
import numpy as np

# ==============================================================================
# 1. SETUP AND LOAD PIPELINE
# ==============================================================================
best_pipeline_path = os.path.join("models", "best_pipeline.pkl")

if not os.path.exists(best_pipeline_path):
    print("[ERROR] Trained pipeline not found. Please train models first using: python main.py")
    exit(1)

print("="*80)
print(" STUDENT PERFORMANCE PREDICTOR - SEQUENTIAL INFERENCE ENGINE")
print("="*80)

# Load unified end-to-end Pipeline
with open(best_pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)
    
print("\n[SUCCESS] Loaded unified end-to-end best-performing ML pipeline successfully.")

# ==============================================================================
# 2. SEQUENTIAL USER INGESTION & INTERACTIVE FLAT VALIDATION
# ==============================================================================

# --- Gender Ingestion ---
gender_options = ['female', 'male']
print("\nSelect Student's Gender:")
for idx, opt in enumerate(gender_options, 1):
    print(f"  {idx}. {opt}")
while True:
    try:
        choice = input("Enter option number: ").strip()
        idx_choice = int(choice) - 1
        if 0 <= idx_choice < len(gender_options):
            gender = gender_options[idx_choice]
            break
        print(f"Invalid option. Enter a number between 1 and {len(gender_options)}.")
    except ValueError:
        print("Please enter a valid number.")

# --- Race / Ethnicity Ingestion ---
race_options = ['group A', 'group B', 'group C', 'group D', 'group E']
print("\nSelect Student's Race/Ethnicity:")
for idx, opt in enumerate(race_options, 1):
    print(f"  {idx}. {opt}")
while True:
    try:
        choice = input("Enter option number: ").strip()
        idx_choice = int(choice) - 1
        if 0 <= idx_choice < len(race_options):
            race = race_options[idx_choice]
            break
        print(f"Invalid option. Enter a number between 1 and {len(race_options)}.")
    except ValueError:
        print("Please enter a valid number.")

# --- Parental Level of Education Ingestion ---
edu_options = [
    'some high school', 'high school', 'some college', 
    "associate's degree", "bachelor's degree", "master's degree"
]
print("\nSelect Parent's Level of Education:")
for idx, opt in enumerate(edu_options, 1):
    print(f"  {idx}. {opt}")
while True:
    try:
        choice = input("Enter option number: ").strip()
        idx_choice = int(choice) - 1
        if 0 <= idx_choice < len(edu_options):
            edu = edu_options[idx_choice]
            break
        print(f"Invalid option. Enter a number between 1 and {len(edu_options)}.")
    except ValueError:
        print("Please enter a valid number.")

# --- Lunch Type Ingestion ---
lunch_options = ['standard', 'free/reduced']
print("\nSelect Lunch Type:")
for idx, opt in enumerate(lunch_options, 1):
    print(f"  {idx}. {opt}")
while True:
    try:
        choice = input("Enter option number: ").strip()
        idx_choice = int(choice) - 1
        if 0 <= idx_choice < len(lunch_options):
            lunch = lunch_options[idx_choice]
            break
        print(f"Invalid option. Enter a number between 1 and {len(lunch_options)}.")
    except ValueError:
        print("Please enter a valid number.")

# --- Test Preparation Course Ingestion ---
prep_options = ['none', 'completed']
print("\nSelect Test Preparation Course:")
for idx, opt in enumerate(prep_options, 1):
    print(f"  {idx}. {opt}")
while True:
    try:
        choice = input("Enter option number: ").strip()
        idx_choice = int(choice) - 1
        if 0 <= idx_choice < len(prep_options):
            prep = prep_options[idx_choice]
            break
        print(f"Invalid option. Enter a number between 1 and {len(prep_options)}.")
    except ValueError:
        print("Please enter a valid number.")

# --- Math Score Ingestion ---
print("\nEnter student's Math Score (0-100):")
while True:
    try:
        val = float(input("Enter score: ").strip())
        if 0 <= val <= 100:
            math_score = val
            break
        print("Score must be between 0 and 100.")
    except ValueError:
        print("Please enter a valid numeric value.")

# ==============================================================================
# 3. DATAFRAME CONSTRUCT & PREDICTION INFERENCE
# ==============================================================================

# Construct raw DataFrame with exact feature names matching the ColumnTransformer preprocessor
input_df = pd.DataFrame([{
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': edu,
    'lunch': lunch,
    'test preparation course': prep,
    'math score': math_score
}])

print("\n" + "-"*40)
print("Input Features Summary:")
print(f"  Gender:             {gender}")
print(f"  Race/Ethnicity:     {race}")
print(f"  Parental Education: {edu}")
print(f"  Lunch:              {lunch}")
print(f"  Prep Course:        {prep}")
print(f"  Math Score:         {math_score}")
print("-"*40)

# Run end-to-end prediction (pipeline handles One-Hot Encoding and Standard Scaling automatically)
prediction = pipeline.predict(input_df)[0]

# Calculate prediction probabilities/confidence level
probability = None
if hasattr(pipeline, "predict_proba"):
    probs = pipeline.predict_proba(input_df)[0]
    probability = probs[1]

# ==============================================================================
# 4. PREDICTION RESULT VISUALIZATION
# ==============================================================================
print("\n" + "="*50)
print(" ML PREDICTION INFERENCE RESULT")
print("="*50)

if prediction == 1:
    print("Result: PASS")
    if probability is not None:
        print(f"Confidence (Probability of Passing): {probability*100:.2f}%")
else:
    print("Result: FAIL")
    if probability is not None:
        print(f"Confidence (Probability of Failing): {(1 - probability)*100:.2f}%")
        
print("="*50)
print("\nInference complete!")
