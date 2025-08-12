"""
Final simplified training script for EEG mental state classifier.
Avoids ICA bottleneck and works with the correct API signatures.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import custom modules
from src.data_processing.data_loader import EEGDataLoader
from src.models.classifier import MentalStateClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a simple feature extraction function to replace the complex one
def extract_basic_features(epochs):
    """Extract basic features from epochs without relying on complex feature extractor."""
    features = {}
    
    # Get data
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Average across epochs if multiple epochs
    if data.shape[0] > 1:
        data = np.mean(data, axis=0)  # Shape: (n_channels, n_times)
        
    # Compute basic time-domain features
    for ch_idx in range(data.shape[0]):
        ch_data = data[ch_idx]
        
        # Basic statistical features
        features[f'mean_ch{ch_idx+1}'] = np.mean(ch_data)
        features[f'std_ch{ch_idx+1}'] = np.std(ch_data)
        features[f'min_ch{ch_idx+1}'] = np.min(ch_data)
        features[f'max_ch{ch_idx+1}'] = np.max(ch_data)
        features[f'range_ch{ch_idx+1}'] = np.max(ch_data) - np.min(ch_data)
        
    return features

def main():
    """Run a simplified training pipeline without ICA and with fixed feature extraction."""
    # Set parameters
    data_path = 'data/synthetic_eeg_expanded'
    model_type = 'svm'
    output_dir = 'src/models/saved_models'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"final_mental_state_classifier_{model_type}.joblib")
    
    # Track timing
    start_time = time.time()
    
    # 1. Load data
    logger.info("Loading expanded synthetic EEG dataset...")
    data_loader = EEGDataLoader(data_path)
    subjects = data_loader.subjects
    logger.info(f"Found data for {len(subjects)} subjects")
    
    all_features = []
    all_labels = []
    
    # Process a limited number of subjects for quick results
    max_subjects = 5  # Start with just 5 subjects for quick results
    subjects_to_process = subjects[:max_subjects]
    
    for subject_id in subjects_to_process:
        try:
            # Process focused data (condition 1)
            logger.info(f"Processing focused data for {subject_id}...")
            
            # Load raw data
            focused_raw = data_loader.load_raw_data(subject_id, condition='focused')
            
            # Simple filtering without ICA
            logger.info("  Applying basic filtering...")
            focused_filtered = focused_raw.copy().filter(l_freq=0.5, h_freq=45.0, 
                                                         picks='eeg', method='fir', phase='zero')
            
            # Create epochs directly
            logger.info("  Creating epochs...")
            sample_rate = int(focused_filtered.info['sfreq'])
            n_samples = len(focused_filtered.times)
            event_spacing = sample_rate  # 1 event per second
            
            # Create fewer events for faster processing (every 2 seconds)
            focused_events = np.array([
                [i, 0, 1] for i in range(0, n_samples, event_spacing*2)
                if i + int(0.5 * sample_rate) < n_samples
            ])
            
            if len(focused_events) == 0:
                logger.warning(f"No valid events created for {subject_id} focused condition")
                continue
                
            focused_epochs = mne.Epochs(focused_filtered, focused_events, tmin=-0.2, tmax=0.5,
                                   baseline=(None, 0), preload=True, verbose=False)
            
            # Extract features using simplified method
            logger.info("  Extracting features...")
            focused_features = extract_basic_features(focused_epochs)
            
            # Process relaxed data (condition 2) 
            logger.info(f"Processing relaxed data for {subject_id}...")
            
            # Load raw data
            relaxed_raw = data_loader.load_raw_data(subject_id, condition='relaxed')
            
            # Simple filtering without ICA
            logger.info("  Applying basic filtering...")
            relaxed_filtered = relaxed_raw.copy().filter(l_freq=0.5, h_freq=45.0, 
                                                         picks='eeg', method='fir', phase='zero')
            
            # Create epochs directly
            logger.info("  Creating epochs...")
            sample_rate = int(relaxed_filtered.info['sfreq'])
            n_samples = len(relaxed_filtered.times)
            
            # Create fewer events for faster processing (every 2 seconds)
            relaxed_events = np.array([
                [i, 0, 1] for i in range(0, n_samples, event_spacing*2)
                if i + int(0.5 * sample_rate) < n_samples
            ])
            
            if len(relaxed_events) == 0:
                logger.warning(f"No valid events created for {subject_id} relaxed condition")
                continue
                
            relaxed_epochs = mne.Epochs(relaxed_filtered, relaxed_events, tmin=-0.2, tmax=0.5,
                                   baseline=(None, 0), preload=True, verbose=False)
            
            # Extract features using simplified method
            logger.info("  Extracting features...")
            relaxed_features = extract_basic_features(relaxed_epochs)
            
            # Add features to dataset
            focused_features_df = pd.DataFrame([focused_features])
            relaxed_features_df = pd.DataFrame([relaxed_features])
            
            focused_features_df['label'] = 1  # 1 = focused
            relaxed_features_df['label'] = 0  # 0 = relaxed
            
            all_features.append(focused_features_df)
            all_features.append(relaxed_features_df)
                
            logger.info(f"Successfully processed subject {subject_id}")
                
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            continue
    
    # 3. Combine features and train model
    if not all_features:
        logger.error("No features extracted. Check preprocessing steps and data.")
        return
        
    logger.info("Combining features and preparing dataset...")
    features_df = pd.concat(all_features, ignore_index=True)
    X = features_df.drop('label', axis=1)
    y = features_df['label'].values
    
    logger.info(f"Final dataset: {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # 4. Train and evaluate model
    logger.info("Training model...")
    classifier = MentalStateClassifier(model_type=model_type)
    
    # Use the correct method signature - train_and_evaluate only accepts X, y, test_size
    result = classifier.train_and_evaluate(X, y, test_size=0.2)
    
    # After training, manually save the model
    model_path = os.path.join(output_dir, f"final_mental_state_classifier_{model_type}.joblib")
    classifier.save_model(model_path)
    
    # Extract metrics from result
    accuracy = result.get('accuracy', 0.0)
    precision = result.get('precision', 0.0)
    recall = result.get('recall', 0.0)
    f1 = result.get('f1', 0.0)
    
    # Fix the confusion matrix - convert to numpy array if it's a list
    confusion_mat = result.get('confusion_matrix', np.array([[0, 0], [0, 0]]))
    if isinstance(confusion_mat, list):
        confusion_mat = np.array(confusion_mat)
    
    y_true = result.get('y_true', [])
    y_pred = result.get('y_pred', [])
    
    # 5. Report results
    logger.info("\n===== MODEL PERFORMANCE =====")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("===========================\n")
    
    # Create and save confusion matrix visualization
    class_names = ['Relaxed', 'Focused']
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations - ensure confusion_mat is a numpy array
    if confusion_mat.size > 0:  # Check if confusion matrix is not empty
        thresh = confusion_mat.max() / 2.0 if confusion_mat.max() > 0 else 0
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if confusion_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save visualization
    cm_path = os.path.join(output_dir, f"confusion_matrix_final_{model_type}.png")
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'prediction': y_pred
    })
    
    # Try to get probabilities if available
    try:
        y_proba = classifier.predict_proba(X)
        if y_proba is not None and len(y_proba) > 0:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                predictions_df['prob_relaxed'] = y_proba[:, 0]
                predictions_df['prob_focused'] = y_proba[:, 1]
    except Exception as e:
        logger.warning(f"Could not get probability predictions: {str(e)}")
    
    predictions_path = os.path.join(output_dir, f"predictions_final_{model_type}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Detailed predictions saved to {predictions_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
