"""
API server for EEG mental state classifier.
Provides endpoints for EEG data processing and mental state prediction.
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import mne
import tempfile
import joblib
from dotenv import load_dotenv
import logging
import sys
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.preprocessor import EEGPreprocessor
from feature_extraction.brainwave_features import BrainwaveFeatureExtractor
from models.classifier import MentalStateClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Get model path from environment or use default
MODEL_PATH = os.getenv("MODEL_PATH", "../models/saved_models/mental_state_classifier.joblib")

# Load the model (will be lazy-loaded when needed)
_model = None


def get_model():
    """Lazy-load the pre-trained model."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            _model = None
    return _model


class HealthCheck(Resource):
    """Health check endpoint to verify API is running."""
    
    def get(self):
        return {"status": "ok", "message": "API is running"}


class PredictMentalState(Resource):
    """Predict mental state from uploaded EEG data."""
    
    def post(self):
        """
        Process EEG data and return predicted mental state.
        Expects EEG data file in EDF format.
        """
        if 'eeg_file' not in request.files:
            return {"error": "No EEG file provided"}, 400
            
        file = request.files['eeg_file']
        if file.filename == '':
            return {"error": "No file selected"}, 400
            
        # Save the uploaded file to a temporary file
        try:
            with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp:
                temp_name = temp.name
                file.save(temp_name)
                
            # Load the EEG data
            try:
                raw = mne.io.read_raw_edf(temp_name, preload=True)
            except Exception as e:
                logger.error(f"Error reading EEG file: {e}")
                return {"error": f"Error reading EEG file: {str(e)}"}, 400
                
            # Preprocess the data
            preprocessor = EEGPreprocessor()
            try:
                raw_processed = preprocessor.preprocess(raw)
            except Exception as e:
                logger.error(f"Error preprocessing EEG data: {e}")
                return {"error": f"Error preprocessing EEG data: {str(e)}"}, 500
                
            # Extract features
            feature_extractor = BrainwaveFeatureExtractor(sfreq=raw.info['sfreq'])
            try:
                features_df = feature_extractor.extract_features(raw_processed)
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                return {"error": f"Error extracting features: {str(e)}"}, 500
                
            # Load the model
            model = get_model()
            if model is None:
                return {"error": "Model not available"}, 503
                
            # Make prediction
            try:
                prediction_result = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)
                
                # Map prediction to mental state labels
                state_labels = {0: "relaxed", 1: "focused"}
                predicted_state = state_labels.get(prediction_result[0], "unknown")
                
                # Format probabilities
                probabilities = {
                    "relaxed": float(prediction_proba[0][0]),
                    "focused": float(prediction_proba[0][1])
                }
                
                return {
                    "predicted_state": predicted_state,
                    "confidence": float(max(prediction_proba[0])),
                    "probabilities": probabilities,
                    "features": features_df.to_dict(orient="records")[0]
                }
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                return {"error": f"Error making prediction: {str(e)}"}, 500
        finally:
            # Clean up the temporary file
            if 'temp_name' in locals():
                try:
                    os.unlink(temp_name)
                except:
                    pass


class ModelInfo(Resource):
    """Get information about the loaded model."""
    
    def get(self):
        model = get_model()
        if model is None:
            return {"error": "Model not available"}, 503
            
        try:
            return {
                "model_type": model.model_type,
                "feature_count": model.feature_names_in_.size if hasattr(model, "feature_names_in_") else "unknown",
                "performance": {
                    "accuracy": model.metrics.get("accuracy", "unknown"),
                    "f1_score": model.metrics.get("f1", "unknown"),
                    "precision": model.metrics.get("precision", "unknown"),
                    "recall": model.metrics.get("recall", "unknown")
                } if hasattr(model, "metrics") else "unknown"
            }
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            return {"error": f"Error retrieving model info: {str(e)}"}, 500


# Register API endpoints
api.add_resource(HealthCheck, '/health')
api.add_resource(PredictMentalState, '/predict')
api.add_resource(ModelInfo, '/model-info')


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
