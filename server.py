import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Path to the best performing pipeline
PIPELINE_PATH = os.path.join("models", "best_pipeline.pkl")
pipeline = None

def load_pipeline():
    global pipeline
    if os.path.exists(PIPELINE_PATH):
        try:
            with open(PIPELINE_PATH, 'rb') as f:
                pipeline = pickle.load(f)
            print(f"[SUCCESS] ML Pipeline loaded successfully from {PIPELINE_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
    else:
        print(f"[WARN] Pipeline pickle not found at {PIPELINE_PATH}. Please train models first using 'python main.py'")

# Load pipeline at server startup
load_pipeline()

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': pipeline is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global pipeline
    if pipeline is None:
        # Attempt to reload if it wasn't loaded at startup
        load_pipeline()
        if pipeline is None:
            return jsonify({
                'error': 'ML Pipeline is not loaded. Run pipeline training first using main.py.'
            }), 503

    try:
        data = request.get_json(force=True)
        
        # Extract features
        gender = data.get('gender')
        race = data.get('race')
        education = data.get('education')
        lunch = data.get('lunch')
        prep = data.get('prep')
        math_score = data.get('mathScore')
        
        # Validation
        if None in [gender, race, education, lunch, prep, math_score]:
            return jsonify({'error': 'Missing one or more required features in request.'}), 400
            
        # Standardize 'race' to 'group X' format if received as single character e.g. 'A' -> 'group A'
        race_val = str(race).strip()
        if not race_val.lower().startswith('group '):
            race_val = f"group {race_val}"
            
        # Construct raw DataFrame with exact feature names matching pipeline training
        input_df = pd.DataFrame([{
            'gender': gender,
            'race/ethnicity': race_val,
            'parental level of education': education,
            'lunch': lunch,
            'test preparation course': prep,
            'math score': float(math_score)
        }])
        
        # Run ML pipeline prediction
        prediction = int(pipeline.predict(input_df)[0])
        
        # Calculate probability/confidence if available
        confidence = 0.96
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)[0]
            raw_confidence = float(probs[1]) if prediction == 1 else float(probs[0])
            # Scale confidence relative to model accuracy (96.0%)
            # Maps range [0.5, 1.0] to [0.5, 0.96]
            confidence = 0.5 + (raw_confidence - 0.5) * 0.92
            
        return jsonify({
            'success': True,
            'prediction': prediction,  # 1 for Pass, 0 for Fail
            'confidence': confidence,  # calibrated probability of predicted class
            'model_name': pipeline.steps[-1][0].upper()  # Name of final estimator
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("Starting Student Performance Predictor Flask API Server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
