"""
Generate an expanded synthetic EEG dataset with more subjects and longer recordings
for improved model training and evaluation.
"""

import os
import logging
import numpy as np
from pathlib import Path
from src.data_processing.synthetic_generator import SyntheticEEGGenerator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate expanded synthetic dataset."""
    # Set parameters for expanded dataset
    output_dir = 'data/synthetic_eeg_expanded'
    n_subjects = 30  # Triple the number of subjects (from 10 to 30)
    conditions = ['focused', 'relaxed']
    duration = 60.0  # Doubled duration (from 30 to 60 seconds)
    sfreq = 500  # Sampling frequency in Hz
    random_seed = 42
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating expanded dataset with {n_subjects} subjects, "
                f"{len(conditions)} conditions, and {duration}s recordings")
    
    # Initialize synthetic EEG generator
    generator = SyntheticEEGGenerator(sfreq=sfreq, random_state=random_seed)
    
    # Generate the dataset
    generator.generate_dataset(
        output_dir=output_dir,
        n_subjects=n_subjects,
        conditions=conditions,
        duration=duration
    )
    
    logger.info("Expanded synthetic dataset generation complete")
    logger.info(f"Dataset saved to {os.path.abspath(output_dir)}")
    
    # Print summary statistics
    n_files = len(list(Path(output_dir).glob('*.edf')))
    expected_files = n_subjects * len(conditions)
    logger.info(f"Generated {n_files}/{expected_files} expected files")
    
    # Calculate approximate dataset size
    channels = 21
    bytes_per_sample = 8  # 8 bytes for float64
    samples_per_subject = int(sfreq * duration * len(conditions) * channels)
    total_samples = samples_per_subject * n_subjects
    dataset_size_mb = (total_samples * bytes_per_sample) / (1024 * 1024)
    
    logger.info(f"Approximate dataset size: {dataset_size_mb:.2f} MB")
    logger.info(f"Each subject has ~{samples_per_subject:,} samples "
                f"({channels} channels × {duration}s × {sfreq}Hz × {len(conditions)} conditions)")

if __name__ == "__main__":
    main()
