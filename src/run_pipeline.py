"""
Main script to run the complete EEG mental state classification pipeline.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from data_processing.data_loader import EEGDataLoader
from data_processing.preprocessor import EEGPreprocessor
from feature_extraction.brainwave_features import BrainwaveFeatureExtractor
from models.classifier import MentalStateClassifier


def get_data_dir():
    """Get the path to the data directory."""
    project_dir = Path(__file__).resolve().parent.parent
    # Check the expected data directory from wget download
    primary_data_dir = project_dir / "data" / "physionet.org" / "files" / "eegmat" / "1.0.0"
    
    if primary_data_dir.exists():
        return str(primary_data_dir)
    
    # Fallback to just the data directory
    fallback_data_dir = project_dir / "data"
    if fallback_data_dir.exists():
        return str(fallback_data_dir)
    
    raise FileNotFoundError("Data directory not found. Please download the dataset first.")


def create_output_dirs():
    """Create necessary output directories."""
    project_dir = Path(__file__).resolve().parent.parent
    
    # Create directories for model and results
    models_dir = project_dir / "models" / "saved_models"
    results_dir = project_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return str(models_dir), str(results_dir)


def train_model(data_dir, models_dir, subject_ids=None, model_type='svm'):
    """
    Train the mental state classifier on multiple subjects.
    
    Args:
        data_dir: Path to the data directory
        models_dir: Path to save the trained model
        subject_ids: List of subject IDs to use (None = use all available)
        model_type: Type of model to train (svm, random_forest, neural_network)
        
    Returns:
        Path to the saved model
    """
    logger.info(f"Training {model_type} model using data from {data_dir}")
    
    # Initialize data loader
    data_loader = EEGDataLoader(data_dir)
    
    # Use specified subjects or all available
    if subject_ids is None:
        subject_ids = data_loader.subjects
    else:
        # Make sure all requested subjects are available
        for subject_id in subject_ids:
            if subject_id not in data_loader.subjects:
                logger.warning(f"Subject {subject_id} not found in dataset")
    
    logger.info(f"Using subjects: {subject_ids}")
    
    # Prepare data from all subjects
    all_features = []
    
    for subject_id in subject_ids:
        logger.info(f"Processing subject {subject_id}")
        
        # Load both conditions for the subject
        try:
            arithmetic_raw = data_loader.load_raw_data(subject_id, "arithmetic")
            baseline_raw = data_loader.load_raw_data(subject_id, "baseline")
            
            if arithmetic_raw is None or baseline_raw is None:
                logger.warning(f"Missing data for subject {subject_id}, skipping")
                continue
            
            # Preprocess the data
            preprocessor = EEGPreprocessor()
            arithmetic_processed = preprocessor.preprocess(arithmetic_raw)
            baseline_processed = preprocessor.preprocess(baseline_raw)
            
            # Extract features
            feature_extractor = BrainwaveFeatureExtractor(sfreq=arithmetic_raw.info['sfreq'])
            
            arithmetic_features = feature_extractor.extract_features(arithmetic_processed)
            arithmetic_features['label'] = 1  # Focused (mental arithmetic)
            arithmetic_features['subject_id'] = subject_id
            
            baseline_features = feature_extractor.extract_features(baseline_processed)
            baseline_features['label'] = 0  # Relaxed (baseline)
            baseline_features['subject_id'] = subject_id
            
            # Combine features for this subject
            subject_features = pd.concat([arithmetic_features, baseline_features], ignore_index=True)
            all_features.append(subject_features)
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {e}")
            continue
    
    # Combine all subjects
    if not all_features:
        raise ValueError("No valid data was processed. Check the dataset and errors.")
        
    all_features_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"Total dataset: {len(all_features_df)} samples, {all_features_df.shape[1]-2} features")
    
    # Shuffle the data
    all_features_df = all_features_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split features and labels
    X = all_features_df.drop(['label', 'subject_id'], axis=1)
    y = all_features_df['label']
    
    # Initialize and train classifier
    classifier = MentalStateClassifier(model_type=model_type, random_state=42)
    
    # Train and evaluate the model
    metrics = classifier.train_and_evaluate(X, y, test_size=0.3)
    logger.info(f"Model performance: {metrics}")
    
    # Save the model
    model_path = os.path.join(models_dir, "mental_state_classifier.joblib")
    classifier.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def predict_mental_state(model_path, data_dir, subject_id, condition):
    """
    Predict mental state for a specific subject and condition.
    
    Args:
        model_path: Path to the trained model
        data_dir: Path to the data directory
        subject_id: Subject ID to predict for
        condition: Condition to predict ('arithmetic' or 'baseline')
        
    Returns:
        Prediction and probabilities
    """
    logger.info(f"Predicting for subject {subject_id}, condition: {condition}")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load data
    data_loader = EEGDataLoader(data_dir)
    
    if subject_id not in data_loader.subjects:
        logger.error(f"Subject {subject_id} not found in dataset")
        return None
    
    # Load and preprocess the data
    raw = data_loader.load_raw_data(subject_id, condition)
    if raw is None:
        logger.error(f"Failed to load {condition} data for subject {subject_id}")
        return None
    
    preprocessor = EEGPreprocessor()
    raw_processed = preprocessor.preprocess(raw)
    
    # Extract features
    feature_extractor = BrainwaveFeatureExtractor(sfreq=raw.info['sfreq'])
    features_df = feature_extractor.extract_features(raw_processed)
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    result = {
        'subject_id': subject_id,
        'condition': condition,
        'prediction': int(prediction),
        'state': "Focused (Mental Arithmetic)" if prediction == 1 else "Relaxed (Baseline)",
        'probability_relaxed': float(probabilities[0]),
        'probability_focused': float(probabilities[1])
    }
    
    logger.info(f"Prediction: {result['state']} (confidence: {max(probabilities):.2f})")
    return result


def visualize_subject_data(data_dir, subject_id, results_dir):
    """
    Create visualizations for a subject's EEG data.
    
    Args:
        data_dir: Path to the data directory
        subject_id: Subject ID to visualize
        results_dir: Directory to save visualizations
        
    Returns:
        Paths to created visualization files
    """
    logger.info(f"Creating visualizations for subject {subject_id}")
    
    # Load data
    data_loader = EEGDataLoader(data_dir)
    
    if subject_id not in data_loader.subjects:
        logger.error(f"Subject {subject_id} not found in dataset")
        return []
    
    visualization_paths = []
    
    try:
        # Load both conditions
        arithmetic_raw = data_loader.load_raw_data(subject_id, "arithmetic")
        baseline_raw = data_loader.load_raw_data(subject_id, "baseline")
        
        if arithmetic_raw is None or baseline_raw is None:
            logger.error(f"Missing data for subject {subject_id}")
            return []
        
        # Preprocess the data
        preprocessor = EEGPreprocessor()
        arithmetic_processed = preprocessor.preprocess(arithmetic_raw)
        baseline_processed = preprocessor.preprocess(baseline_raw)
        
        # Sample raw and processed signals
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        # Original signals
        t_orig = np.linspace(0, 5, int(5 * arithmetic_raw.info['sfreq']))
        ch_idx = 0  # Use first channel for example
        ch_name = arithmetic_raw.ch_names[ch_idx]
        
        axes[0, 0].plot(t_orig, arithmetic_raw[ch_idx, :len(t_orig)][0][0], 'b')
        axes[0, 0].set_title(f'Original Signal - Mental Arithmetic - {ch_name}')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude (µV)')
        
        axes[0, 1].plot(t_orig, baseline_raw[ch_idx, :len(t_orig)][0][0], 'r')
        axes[0, 1].set_title(f'Original Signal - Baseline - {ch_name}')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude (µV)')
        
        # Processed signals
        axes[1, 0].plot(t_orig, arithmetic_processed[ch_idx, :len(t_orig)][0][0], 'b')
        axes[1, 0].set_title(f'Processed Signal - Mental Arithmetic - {ch_name}')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude (µV)')
        
        axes[1, 1].plot(t_orig, baseline_processed[ch_idx, :len(t_orig)][0][0], 'r')
        axes[1, 1].set_title(f'Processed Signal - Baseline - {ch_name}')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude (µV)')
        
        plt.tight_layout()
        
        # Save figure
        raw_signals_path = os.path.join(results_dir, f"subject_{subject_id}_signals.png")
        plt.savefig(raw_signals_path)
        visualization_paths.append(raw_signals_path)
        
        # Extract features
        feature_extractor = BrainwaveFeatureExtractor(sfreq=arithmetic_raw.info['sfreq'])
        
        arithmetic_features = feature_extractor.extract_features(arithmetic_processed)
        baseline_features = feature_extractor.extract_features(baseline_processed)
        
        # Plot band powers
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        for i, band in enumerate(bands):
            # Find all columns with this band's power
            arith_cols = [c for c in arithmetic_features.columns if f"{band}_power" in c]
            base_cols = [c for c in baseline_features.columns if f"{band}_power" in c]
            
            # Extract the channel name from column names
            channels = [c.split('_')[0] for c in arith_cols]
            
            # Get the power values
            arith_powers = arithmetic_features[arith_cols].values[0]
            base_powers = baseline_features[base_cols].values[0]
            
            # Bar plot
            x = np.arange(len(channels))
            width = 0.35
            
            axes[i].bar(x - width/2, arith_powers, width, label='Mental Arithmetic')
            axes[i].bar(x + width/2, base_powers, width, label='Baseline')
            
            axes[i].set_title(f'{band.capitalize()} Band Power')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(channels, rotation=90)
            
            if i == 0:
                axes[i].set_ylabel('Power')
                axes[i].legend()
        
        plt.tight_layout()
        
        # Save figure
        band_powers_path = os.path.join(results_dir, f"subject_{subject_id}_band_powers.png")
        plt.savefig(band_powers_path)
        visualization_paths.append(band_powers_path)
        
        logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return []


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='EEG Mental State Classification Pipeline')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--subject', type=str, default='01', help='Subject ID for prediction')
    parser.add_argument('--condition', type=str, choices=['arithmetic', 'baseline'], 
                        default='arithmetic', help='Condition for prediction')
    parser.add_argument('--model-type', type=str, choices=['svm', 'random_forest', 'neural_network'],
                        default='svm', help='Model type for training')
    
    args = parser.parse_args()
    
    try:
        # Get directories
        data_dir = get_data_dir()
        models_dir, results_dir = create_output_dirs()
        
        # Default model path
        model_path = os.path.join(models_dir, "mental_state_classifier.joblib")
        
        # Training
        if args.train:
            model_path = train_model(data_dir, models_dir, model_type=args.model_type)
        
        # Prediction
        if args.predict:
            if not os.path.exists(model_path):
                logger.warning("No trained model found. Training a model first...")
                model_path = train_model(data_dir, models_dir)
            
            result = predict_mental_state(model_path, data_dir, args.subject, args.condition)
            
            if result:
                print("\n===== PREDICTION RESULT =====")
                print(f"Subject: {result['subject_id']}")
                print(f"Condition: {result['condition']}")
                print(f"Predicted mental state: {result['state']}")
                print(f"Confidence: {max(result['probability_relaxed'], result['probability_focused']):.2f}")
                print("============================\n")
        
        # Visualization
        if args.visualize:
            paths = visualize_subject_data(data_dir, args.subject, results_dir)
            
            if paths:
                print("\n===== VISUALIZATIONS =====")
                for path in paths:
                    print(f"Created: {path}")
                print("=========================\n")
        
        # If no actions specified, run everything
        if not (args.train or args.predict or args.visualize):
            logger.info("No specific action specified, running complete pipeline")
            
            if not os.path.exists(model_path):
                model_path = train_model(data_dir, models_dir)
            
            # Predict for both conditions
            for condition in ['arithmetic', 'baseline']:
                predict_mental_state(model_path, data_dir, args.subject, condition)
            
            # Create visualizations
            visualize_subject_data(data_dir, args.subject, results_dir)
    
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
