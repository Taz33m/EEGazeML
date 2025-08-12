"""
End-to-end test for EEG mental state classifier.
Tests the complete pipeline from data loading to classification.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing.data_loader import EEGDataLoader
from src.data_processing.preprocessor import EEGPreprocessor
from src.feature_extraction.brainwave_features import BrainwaveFeatureExtractor
from src.models.classifier import MentalStateClassifier


class TestEndToEnd(unittest.TestCase):
    """Test the complete EEG mental state classification pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Set paths
        cls.project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.data_dir = cls.project_dir / "data" / "physionet.org" / "files" / "eegmat" / "1.0.0"
        cls.models_dir = cls.project_dir / "src" / "models" / "saved_models"
        cls.results_dir = cls.project_dir / "tests" / "results"
        
        # Create directories if they don't exist
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Constants
        cls.test_subject = "Subject01"  # Use consistent subject for testing
        cls.model_path = cls.models_dir / "mental_state_classifier.joblib"
        
        logger.info(f"Using data directory: {cls.data_dir}")
        
        # Check if dataset exists
        if not cls.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {cls.data_dir}")
            
        files = list(cls.data_dir.glob(f"{cls.test_subject}_*.edf"))
        if not files:
            raise FileNotFoundError(f"No data files found for {cls.test_subject}")
            
        logger.info(f"Found test files: {[f.name for f in files]}")
    
    def test_data_loading(self):
        """Test data loading from EDF files."""
        logger.info("Testing data loading...")
        
        # Initialize data loader
        data_loader = EEGDataLoader(str(self.data_dir))
        
        # Check that subjects were discovered
        self.assertTrue(len(data_loader.subjects) > 0, "No subjects discovered")
        logger.info(f"Found {len(data_loader.subjects)} subjects")
        
        # Check that test subject exists
        subject_id = self.test_subject.replace("Subject", "")
        self.assertIn(subject_id, data_loader.subjects, f"Test subject {subject_id} not found")
        
        # Load data for arithmetic and baseline conditions
        arithmetic_raw = data_loader.load_raw_data(subject_id, "arithmetic")
        self.assertIsNotNone(arithmetic_raw, "Failed to load arithmetic data")
        logger.info(f"Loaded arithmetic data: {len(arithmetic_raw.times)} samples, "
                    f"{len(arithmetic_raw.ch_names)} channels")
        
        baseline_raw = data_loader.load_raw_data(subject_id, "baseline")
        self.assertIsNotNone(baseline_raw, "Failed to load baseline data")
        logger.info(f"Loaded baseline data: {len(baseline_raw.times)} samples, "
                    f"{len(baseline_raw.ch_names)} channels")
        
        # Create epochs
        epochs_dict = data_loader.create_epochs(subject_id, tmin=-0.2, tmax=0.5)
        self.assertIsNotNone(epochs_dict, "Failed to create epochs")
        logger.info(f"Created epochs for conditions: {list(epochs_dict.keys())}")
    
    def test_preprocessing(self):
        """Test preprocessing of EEG data."""
        logger.info("Testing preprocessing...")
        
        # Load raw data
        data_loader = EEGDataLoader(str(self.data_dir))
        subject_id = self.test_subject.replace("Subject", "")
        raw = data_loader.load_raw_data(subject_id, "arithmetic")
        self.assertIsNotNone(raw, "Failed to load raw data")
        
        # Initialize preprocessor
        preprocessor = EEGPreprocessor()
        
        # Apply preprocessing
        raw_processed = preprocessor.preprocess(raw, apply_notch=True, remove_artifacts=True)
        self.assertIsNotNone(raw_processed, "Failed to preprocess data")
        
        # Check that preprocessing didn't remove all data
        self.assertTrue(len(raw_processed.times) > 0, "Preprocessed data has no time points")
        self.assertTrue(len(raw_processed.ch_names) > 0, "Preprocessed data has no channels")
        
        logger.info(f"Preprocessed data: {len(raw_processed.times)} samples, "
                    f"{len(raw_processed.ch_names)} channels")
    
    def test_feature_extraction(self):
        """Test feature extraction from EEG data."""
        logger.info("Testing feature extraction...")
        
        # Load and preprocess data
        data_loader = EEGDataLoader(str(self.data_dir))
        subject_id = self.test_subject.replace("Subject", "")
        raw = data_loader.load_raw_data(subject_id, "arithmetic")
        self.assertIsNotNone(raw, "Failed to load raw data")
        
        preprocessor = EEGPreprocessor()
        raw_processed = preprocessor.preprocess(raw, apply_notch=True, remove_artifacts=True)
        self.assertIsNotNone(raw_processed, "Failed to preprocess data")
        
        # Initialize feature extractor
        feature_extractor = BrainwaveFeatureExtractor(sfreq=raw.info['sfreq'])
        
        # Extract features
        features_df = feature_extractor.extract_features(raw_processed)
        self.assertIsNotNone(features_df, "Failed to extract features")
        self.assertFalse(features_df.empty, "Feature DataFrame is empty")
        
        # Check that we have expected frequency band features
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            band_cols = [col for col in features_df.columns if f"{band}_power" in col]
            self.assertTrue(len(band_cols) > 0, f"No {band} band features extracted")
        
        logger.info(f"Extracted {features_df.shape[1]} features from EEG data")
        
        # Save a sample of the features for later review
        features_df.head().to_csv(self.results_dir / "sample_features.csv")
    
    def test_model_training(self):
        """Test model training and evaluation."""
        logger.info("Testing model training...")
        
        # Load and preprocess data for both conditions
        data_loader = EEGDataLoader(str(self.data_dir))
        subject_id = self.test_subject.replace("Subject", "")
        
        # Get raw data for both conditions
        arithmetic_raw = data_loader.load_raw_data(subject_id, "arithmetic")
        baseline_raw = data_loader.load_raw_data(subject_id, "baseline")
        self.assertIsNotNone(arithmetic_raw, "Failed to load arithmetic data")
        self.assertIsNotNone(baseline_raw, "Failed to load baseline data")
        
        # Preprocess both conditions
        preprocessor = EEGPreprocessor()
        arithmetic_processed = preprocessor.preprocess(arithmetic_raw)
        baseline_processed = preprocessor.preprocess(baseline_raw)
        
        # Extract features
        feature_extractor = BrainwaveFeatureExtractor(sfreq=arithmetic_raw.info['sfreq'])
        arithmetic_features = feature_extractor.extract_features(arithmetic_processed)
        baseline_features = feature_extractor.extract_features(baseline_processed)
        
        # Add labels: 1 for arithmetic (focused), 0 for baseline (relaxed)
        arithmetic_features['label'] = 1
        baseline_features['label'] = 0
        
        # Combine features
        all_features = pd.concat([arithmetic_features, baseline_features], ignore_index=True)
        
        # Shuffle the data
        all_features = all_features.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split features and labels
        X = all_features.drop('label', axis=1)
        y = all_features['label']
        
        # Initialize and train classifier
        classifier = MentalStateClassifier(model_type='svm', random_state=42)
        
        # Train and evaluate the model
        metrics = classifier.train_and_evaluate(X, y, test_size=0.3)
        
        # Check performance metrics
        self.assertGreater(metrics['accuracy'], 0.5, "Model accuracy is no better than random")
        logger.info(f"Model performance: {metrics}")
        
        # Save the model
        classifier.save_model(str(self.model_path))
        self.assertTrue(os.path.exists(self.model_path), "Model was not saved")
        logger.info(f"Model saved to {self.model_path}")
        
        # Create confusion matrix visualization
        if hasattr(classifier, 'confusion_matrix'):
            cm = classifier.confusion_matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Relaxed', 'Focused'],
                        yticklabels=['Relaxed', 'Focused'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / "confusion_matrix.png")
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from data loading to prediction."""
        logger.info("Testing end-to-end pipeline...")
        
        # Check if model exists, if not, train it
        if not os.path.exists(self.model_path):
            logger.info("Model not found, running training test first")
            self.test_model_training()
        
        # Load the model
        try:
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
        
        # Load new data for prediction
        data_loader = EEGDataLoader(str(self.data_dir))
        # Use a different subject for testing predictions
        test_subject = "Subject02"
        subject_id = test_subject.replace("Subject", "")
        
        # Try both conditions to verify that we get different predictions
        for condition, expected_label in [("arithmetic", 1), ("baseline", 0)]:
            # Load and preprocess data
            raw = data_loader.load_raw_data(subject_id, condition)
            self.assertIsNotNone(raw, f"Failed to load {condition} data")
            
            preprocessor = EEGPreprocessor()
            raw_processed = preprocessor.preprocess(raw)
            
            # Extract features
            feature_extractor = BrainwaveFeatureExtractor(sfreq=raw.info['sfreq'])
            features_df = feature_extractor.extract_features(raw_processed)
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            
            logger.info(f"Condition: {condition} (expected {expected_label})")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Probability: relaxed={probability[0]:.4f}, focused={probability[1]:.4f}")
            
            # Check that the prediction is as expected
            # Note: We use assertIn to allow for some flexibility, as the model might not be perfect
            self.assertEqual(
                prediction, expected_label,
                f"Prediction for {condition} was incorrect"
            )


if __name__ == "__main__":
    unittest.main()
