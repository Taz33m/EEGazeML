"""
End-to-end test for EEG mental state classifier using synthetic EEG data.
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


class TestSyntheticPipeline(unittest.TestCase):
    """Test the complete EEG mental state classification pipeline using synthetic data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Set paths
        cls.project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.data_dir = cls.project_dir / "data" / "synthetic_eeg"
        cls.models_dir = cls.project_dir / "src" / "models" / "saved_models"
        cls.results_dir = cls.project_dir / "tests" / "results"
        
        # Create directories if they don't exist
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Constants - use multiple subjects for more robust testing
        cls.test_subjects = ["Subject01", "Subject02", "Subject03"]
        cls.model_path = cls.models_dir / "synthetic_mental_state_classifier.joblib"
        
        logger.info(f"Using synthetic data directory: {cls.data_dir}")
        
        # Check if dataset exists
        if not cls.data_dir.exists():
            raise FileNotFoundError(f"Synthetic dataset directory not found: {cls.data_dir}")
        
        # Verify we have files for each test subject
        for subject in cls.test_subjects:
            files = list(cls.data_dir.glob(f"{subject}_*.edf"))
            if not files:
                raise FileNotFoundError(f"No data files found for {subject}")
            logger.info(f"Found test files for {subject}: {[f.name for f in files]}")
    
    def test_data_loading(self):
        """Test data loading from synthetic EDF files."""
        logger.info("Testing data loading with synthetic data...")
        
        # Initialize data loader
        data_loader = EEGDataLoader(str(self.data_dir))
        
        # Check that subjects were discovered
        self.assertTrue(len(data_loader.subjects) > 0, "No subjects discovered")
        logger.info(f"Found {len(data_loader.subjects)} subjects")
        
        # Check that test subjects exist
        for subject in self.test_subjects:
            subject_id = subject.replace("Subject", "")
            self.assertIn(subject_id, data_loader.subjects, f"Test subject {subject_id} not found")
        
        # Load data for focused and relaxed conditions
        subject_id = self.test_subjects[0].replace("Subject", "")
        focused_raw = data_loader.load_raw_data(subject_id, "focused")
        self.assertIsNotNone(focused_raw, "Failed to load focused data")
        logger.info(f"Loaded focused data: {len(focused_raw.times)} samples, "
                    f"{len(focused_raw.ch_names)} channels")
        
        relaxed_raw = data_loader.load_raw_data(subject_id, "relaxed")
        self.assertIsNotNone(relaxed_raw, "Failed to load relaxed data")
        logger.info(f"Loaded relaxed data: {len(relaxed_raw.times)} samples, "
                    f"{len(relaxed_raw.ch_names)} channels")
        
        # Create epochs
        epochs_dict = data_loader.create_epochs(subject_id, tmin=-0.2, tmax=0.5)
        self.assertIsNotNone(epochs_dict, "Failed to create epochs")
        logger.info(f"Created epochs for conditions: {list(epochs_dict.keys())}")
    
    def test_preprocessing(self):
        """Test preprocessing of EEG data."""
        logger.info("Testing preprocessing with synthetic data...")
        
        # Load raw data
        data_loader = EEGDataLoader(str(self.data_dir))
        subject_id = self.test_subjects[0].replace("Subject", "")
        raw = data_loader.load_raw_data(subject_id, "focused")
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
        """Test feature extraction from synthetic EEG data."""
        logger.info("Testing feature extraction with synthetic data...")
        
        # Load and preprocess data
        data_loader = EEGDataLoader(str(self.data_dir))
        preprocessor = EEGPreprocessor()
        
        subject_id = self.test_subjects[0].replace("Subject", "")
        raw = data_loader.load_raw_data(subject_id, "focused")
        raw_processed = preprocessor.preprocess(raw)
        
        # Initialize feature extractor
        feature_extractor = BrainwaveFeatureExtractor()
        
        # Extract features
        features = feature_extractor.extract_features(raw_processed)
        self.assertIsNotNone(features, "Failed to extract features")
        self.assertIsInstance(features, pd.DataFrame, "Features should be a DataFrame")
        self.assertTrue(len(features) > 0, "No features were extracted")
        
        logger.info(f"Extracted {len(features.columns)} features: {list(features.columns)}")
        
    def test_model_training_multi_subject(self):
        """Test model training and evaluation with multiple subjects."""
        logger.info("Testing model training with multiple synthetic subjects...")
        
        # Load and preprocess data for multiple subjects
        data_loader = EEGDataLoader(str(self.data_dir))
        preprocessor = EEGPreprocessor()
        feature_extractor = BrainwaveFeatureExtractor()
        
        all_features = []
        
        # Process each test subject and condition
        for subject in self.test_subjects:
            subject_id = subject.replace("Subject", "")
            
            # Process focused condition
            focused_raw = data_loader.load_raw_data(subject_id, "focused")
            focused_processed = preprocessor.preprocess(focused_raw)
            focused_features = feature_extractor.extract_features(focused_processed)
            focused_features['label'] = 1  # 1 for focused
            all_features.append(focused_features)
            
            # Process relaxed condition
            relaxed_raw = data_loader.load_raw_data(subject_id, "relaxed")
            relaxed_processed = preprocessor.preprocess(relaxed_raw)
            relaxed_features = feature_extractor.extract_features(relaxed_processed)
            relaxed_features['label'] = 0  # 0 for relaxed
            all_features.append(relaxed_features)
        
        # Combine all features
        all_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined features from {len(self.test_subjects)} subjects, "
                    f"total samples: {len(all_features)}")
        
        # Shuffle the data
        all_features = all_features.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split features and labels
        X = all_features.drop('label', axis=1)
        y = all_features['label']
        
        # Initialize and train classifier
        classifier = MentalStateClassifier(model_type='svm', random_state=42)
        
        # Train and evaluate the model
        metrics = classifier.train_and_evaluate(X, y, test_size=0.3)
        
        # Check performance metrics - more relaxed thresholds for synthetic data testing
        self.assertGreater(metrics['accuracy'], 0.6, "Model accuracy should be better than 0.6 with synthetic data")
        logger.info(f"Model performance: {metrics}")
        
        # Save the model
        classifier.save_model(str(self.model_path))
        self.assertTrue(os.path.exists(self.model_path), "Model was not saved")
        logger.info(f"Model saved to {self.model_path}")
        
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from data loading to prediction with synthetic data."""
        logger.info("Testing end-to-end pipeline with synthetic data...")
        
        # Run through the entire process
        self.test_data_loading()
        self.test_preprocessing()
        self.test_feature_extraction()
        self.test_model_training_multi_subject()
        
        # Load the saved model
        classifier = MentalStateClassifier()
        classifier.load_model(str(self.model_path))
        
        # Load and preprocess a new subject for prediction
        # Using the last test subject in our list
        data_loader = EEGDataLoader(str(self.data_dir))
        preprocessor = EEGPreprocessor()
        feature_extractor = BrainwaveFeatureExtractor()
        
        subject_id = self.test_subjects[-1].replace("Subject", "")
        raw = data_loader.load_raw_data(subject_id, "focused")  # Predicting on focused data
        raw_processed = preprocessor.preprocess(raw)
        features = feature_extractor.extract_features(raw_processed)
        
        # Make prediction
        prediction = classifier.predict(features)
        self.assertIsNotNone(prediction, "Failed to make prediction")
        logger.info(f"Prediction for {subject_id} focused data: {prediction}")
        
        # Check prediction probability
        proba = classifier.predict_proba(features)
        self.assertIsNotNone(proba, "Failed to get prediction probabilities")
        logger.info(f"Prediction probability: {proba}")
        
        # Success message
        logger.info("End-to-end pipeline test completed successfully with synthetic data!")


if __name__ == "__main__":
    unittest.main()
