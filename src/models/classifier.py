"""
Machine learning model for EEG mental state classification.
Trains and evaluates models to classify mental states from EEG features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Any, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MentalStateClassifier:
    """Classifier for mental states based on EEG features."""
    
    # Dictionary of supported model types and their hyperparameter grids
    MODEL_TYPES = {
        'svm': {
            'classifier': SVC,
            'param_grid': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__kernel': ['rbf', 'linear']
            }
        },
        'random_forest': {
            'classifier': RandomForestClassifier,
            'param_grid': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'neural_network': {
            'classifier': MLPClassifier,
            'param_grid': {
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate': ['constant', 'adaptive']
            }
        }
    }
    
    def __init__(self, model_type: str = 'svm', random_state: int = 42):
        """
        Initialize the mental state classifier.
        
        Args:
            model_type: Type of model to use ('svm', 'random_forest', or 'neural_network')
            random_state: Random seed for reproducibility
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.MODEL_TYPES.keys())}")
        
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.feature_importances_ = None
        self.classes_ = None
        
        # Create the pipeline with preprocessing and the selected classifier
        classifier_class = self.MODEL_TYPES[model_type]['classifier']
        
        if model_type == 'svm':
            classifier_kwargs = {'random_state': random_state, 'probability': True}  # Enable probability prediction for SVM
        elif model_type == 'neural_network':
            classifier_kwargs = {'random_state': random_state}
        else:
            classifier_kwargs = {'random_state': random_state, 'n_jobs': -1}
            
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier_class(**classifier_kwargs))
        ])
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            optimize_hyperparams: bool = True,
            cv_folds: int = 5) -> Dict:
        """
        Train the model on the provided features.
        
        Args:
            X: Features DataFrame
            y: Target labels (0 = baseline/relaxed, 1 = focused/arithmetic)
            optimize_hyperparams: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds for hyperparameter optimization
            
        Returns:
            Dictionary with training metrics
        """
        # Store feature names and classes
        self.feature_names_ = list(X.columns)
        self.classes_ = np.unique(y)
        
        # Check if we have enough data to do an internal validation split
        min_train_samples = 3  # Minimum number of samples needed for meaningful training
        
        if len(X) <= min_train_samples:
            logger.info(f"Dataset too small ({len(X)} samples) for internal validation split. Using all data for training.")
            X_train, y_train = X, y
            X_val, y_val = X, y  # Use training data for validation when too small
        else:
            # Check if stratification is possible
            class_counts = np.bincount(y.astype(int))
            can_stratify = np.all(class_counts >= 2)
            
            # Split data for evaluation
            if can_stratify:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=y
                )
                logger.info("Using stratified validation split within fit method")
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=None
                )
                logger.info("Using regular validation split within fit method (not enough samples per class)")
                # Log the class distribution after split
                train_class_counts = np.bincount(y_train.astype(int)) if len(y_train) > 0 else []
                val_class_counts = np.bincount(y_val.astype(int)) if len(y_val) > 0 else []
                logger.info(f"Validation split - Train class distribution: {train_class_counts}")
                logger.info(f"Validation split - Val class distribution: {val_class_counts}")
        
        logger.info(f"Training {self.model_type} model on {X_train.shape[0]} samples")
        
        # Determine if we can do hyperparameter optimization
        # Need at least 2*cv_folds samples for meaningful cross-validation
        min_samples_for_cv = cv_folds * 2
        
        # Hyperparameter optimization if requested and feasible
        if optimize_hyperparams and len(X_train) >= min_samples_for_cv:
            param_grid = self.MODEL_TYPES[self.model_type]['param_grid']
            
            logger.info(f"Performing hyperparameter optimization with {cv_folds}-fold CV")
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=cv_folds, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            if optimize_hyperparams and len(X_train) < min_samples_for_cv:
                logger.info(f"Dataset too small ({len(X_train)} samples) for {cv_folds}-fold CV. Skipping hyperparameter optimization.")
            
            self.pipeline.fit(X_train, y_train)
        
        # Store feature importances if available
        if hasattr(self.pipeline[-1], 'feature_importances_'):
            self.feature_importances_ = pd.Series(
                self.pipeline[-1].feature_importances_,
                index=self.feature_names_
            ).sort_values(ascending=False)
        
        # Evaluate on validation set
        y_pred = self.pipeline.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred)
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict mental states for new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for mental states.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of class probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self.pipeline[-1], 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Features DataFrame
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        y_pred = self.pipeline.predict(X)
        return self._calculate_metrics(y, y_pred)
    
    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the model and evaluate its performance using a test split.
        
        Args:
            X: Features DataFrame
            y: Target labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if we have enough samples for a meaningful split
        if len(X) <= 2:
            logger.warning(f"Only {len(X)} samples available. Using leave-one-out approach.")
            # For extremely small datasets, use all but one sample for training
            if len(X) == 1:
                # Special case: just use the single sample for both training and testing
                # This is not ideal but allows the pipeline to run for demo purposes
                X_train, y_train = X, y
                X_test, y_test = X, y
                logger.warning("Only 1 sample available! Using same sample for training and testing.")
            else:  # 2 samples
                # Ensure at least one sample from each available class if possible
                unique_classes = np.unique(y)
                if len(unique_classes) > 1:
                    # Try to create balanced train/test split with one sample from each class
                    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
                    train_idx = [class_indices[cls][0] for cls in unique_classes]
                    
                    # Since we have only 2 samples, use one for training and one for testing
                    # Always ensure we have at least one sample in each set
                    X_train, X_test = X.iloc[[0]], X.iloc[[1]]
                    y_train, y_test = y[[0]], y[[1]]
                    logger.info("Created balanced train/test split with one sample for training and one for testing")
                else:
                    # Only one class present, use first sample for training, second for testing
                    X_train, X_test = X.iloc[[0]], X.iloc[[1]]
                    y_train, y_test = y[[0]], y[[1]]
                    logger.info("Only one class present. Using first sample for training, second for testing.")
        else:
            # Check if stratification is possible (need at least 2 samples per class)
            class_counts = np.bincount(y.astype(int))
            can_stratify = np.all(class_counts >= 2)
            
            # Split the data into training and test sets
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=y
                )
                logger.info("Using stratified train-test split")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=None
                )
                logger.info("Using regular train-test split (not enough samples per class for stratification)")
                # Log the class distribution after split
                if len(y_train) > 0 and len(y_test) > 0:
                    train_class_counts = np.bincount(y_train.astype(int)) if len(np.unique(y_train)) > 0 else []
                    test_class_counts = np.bincount(y_test.astype(int)) if len(np.unique(y_test)) > 0 else []
                    logger.info(f"Train class distribution: {train_class_counts}")
                    logger.info(f"Test class distribution: {test_class_counts}")
        
        logger.info(f"Split data into {X_train.shape[0]} training and {X_test.shape[0]} test samples")
        
        # Check if training data has at least two classes
        unique_train_classes = np.unique(y_train)
        if len(unique_train_classes) < 2:
            logger.warning("Training data has only one class. Adding a synthetic sample from another class.")
            # Create a synthetic sample from another class
            available_classes = np.unique(y)
            missing_class = [cls for cls in available_classes if cls not in unique_train_classes]
            
            if missing_class:
                # If there's a class in the full dataset that's missing from training,
                # add a synthetic sample from that class
                new_class = missing_class[0]
            else:
                # Otherwise create a new synthetic class
                new_class = max(available_classes) + 1 if len(available_classes) > 0 else 1
            
            # Create synthetic sample (average of existing samples or copy with small noise)
            if len(X) > 1:
                # Create average of existing samples with small noise
                synthetic_x = X.iloc[0:1].copy()  # Copy the first sample as a template
                for col in synthetic_x.columns:
                    synthetic_x[col] = synthetic_x[col] * 1.05  # Add small variation
            else:
                # Just copy the single sample with some noise
                synthetic_x = X.iloc[0:1].copy()
                for col in synthetic_x.columns:
                    synthetic_x[col] = synthetic_x[col] * 1.1  # Add more variation
            
            # Add the synthetic sample to training data
            X_train = pd.concat([X_train, synthetic_x])
            y_train = np.append(y_train, new_class)
            
            logger.warning(f"Added synthetic sample with class {new_class} to enable training")
        
        # Train the model on the training data
        self.fit(X_train, y_train)
        
        # Evaluate on the test data
        metrics = self.evaluate(X_test, y_test)
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.pipeline, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names_,
            'classes': self.classes_.tolist(),
            'feature_importances': self.feature_importances_.to_dict() if self.feature_importances_ is not None else None
        }
        
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'MentalStateClassifier':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded MentalStateClassifier instance
        """
        # Load the pipeline
        pipeline = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.joblib'
        metadata = joblib.load(metadata_path)
        
        # Create a new instance with the correct model type
        instance = cls(model_type=metadata['model_type'])
        
        # Set the loaded pipeline and metadata
        instance.pipeline = pipeline
        instance.feature_names_ = metadata['feature_names']
        instance.classes_ = np.array(metadata['classes'])
        
        if metadata['feature_importances'] is not None:
            instance.feature_importances_ = pd.Series(metadata['feature_importances'])
        
        logger.info(f"Model loaded from {model_path}")
        return instance
