"""
Synthetic EEG Data Generator

This module provides functionality to generate realistic synthetic EEG data for
training and testing the mental state classifier. It creates simulated EEG signals
that mimic the characteristics of real brain activity during different mental states.
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, List, Tuple, Union
import logging
from scipy import signal
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SyntheticEEGGenerator:
    """
    Generates synthetic EEG data that mimics real brainwave patterns
    for different mental states (focused and relaxed).
    """
    
    # Define standard 10-20 EEG channel names
    CHANNEL_NAMES = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
        'T3', 'C3', 'Cz', 'C4', 'T4', 
        'T5', 'P3', 'Pz', 'P4', 'T6', 
        'O1', 'O2', 'A1', 'A2'
    ]
    
    # Define frequency bands in Hz
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Mental state profiles - relative power in different bands
    # These are approximate values based on literature
    MENTAL_STATE_PROFILES = {
        'focused': {
            'delta': 0.1,  # Lower delta during focus
            'theta': 0.15, # Moderate theta 
            'alpha': 0.2,  # Moderate alpha
            'beta': 0.4,   # Higher beta during focus/concentration
            'gamma': 0.15  # Some gamma activity during complex cognition
        },
        'relaxed': {
            'delta': 0.15, # Slightly higher delta during relaxation
            'theta': 0.2,  # Higher theta during relaxation
            'alpha': 0.4,  # Much higher alpha during relaxation
            'beta': 0.15,  # Lower beta during relaxation
            'gamma': 0.1   # Lower gamma during relaxation
        }
    }
    
    # Channel-specific modifiers to add topographic variation
    # Based on general patterns of brain activity
    CHANNEL_MODIFIERS = {
        # Frontal channels - executive function
        'Fp1': {'beta': 1.2, 'gamma': 1.1},
        'Fp2': {'beta': 1.2, 'gamma': 1.1},
        'F7': {'beta': 1.1},
        'F3': {'beta': 1.1},
        'Fz': {'beta': 1.1},
        'F4': {'beta': 1.1},
        'F8': {'beta': 1.1},
        
        # Central channels - sensorimotor 
        'C3': {'beta': 1.1, 'mu': 1.3},
        'Cz': {'beta': 1.1, 'mu': 1.3},
        'C4': {'beta': 1.1, 'mu': 1.3},
        
        # Temporal channels
        'T3': {'theta': 1.1},
        'T4': {'theta': 1.1},
        'T5': {'theta': 1.1},
        'T6': {'theta': 1.1},
        
        # Parietal channels - sensory integration
        'P3': {'alpha': 1.2},
        'Pz': {'alpha': 1.2},
        'P4': {'alpha': 1.2},
        
        # Occipital channels - visual processing, strong alpha
        'O1': {'alpha': 1.4},
        'O2': {'alpha': 1.4},
    }
    
    def __init__(self, 
                 sfreq: float = 500.0, 
                 random_state: int = 42):
        """
        Initialize the synthetic EEG generator.
        
        Args:
            sfreq: Sampling frequency in Hz
            random_state: Random seed for reproducibility
        """
        self.sfreq = sfreq
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def generate_band_signal(self, 
                             band: str, 
                             duration: float, 
                             amplitude: float) -> np.ndarray:
        """
        Generate a signal within a specific frequency band.
        
        Args:
            band: Frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
            duration: Signal duration in seconds
            amplitude: Base amplitude for the signal
            
        Returns:
            1D numpy array with the generated signal
        """
        # Get band frequency range
        low_freq, high_freq = self.FREQ_BANDS[band]
        
        # Number of samples
        n_samples = int(duration * self.sfreq)
        
        # Create time array
        t = np.arange(n_samples) / self.sfreq
        
        # Generate multiple oscillations within the band and sum them
        signal = np.zeros(n_samples)
        
        # Number of distinct oscillations to generate within this band
        n_oscillations = int((high_freq - low_freq) * 2)
        n_oscillations = max(2, min(n_oscillations, 10))  # At least 2, at most 10
        
        for _ in range(n_oscillations):
            # Random frequency within the band
            freq = self.rng.uniform(low_freq, high_freq)
            
            # Random phase
            phase = self.rng.uniform(0, 2 * np.pi)
            
            # Random sub-amplitude (variation within the band)
            sub_amp = self.rng.uniform(0.5, 1.0)
            
            # Generate oscillation and add to signal
            oscillation = sub_amp * np.sin(2 * np.pi * freq * t + phase)
            signal += oscillation
        
        # Normalize and scale by desired amplitude
        signal = signal / np.max(np.abs(signal)) * amplitude
        
        return signal
        
    def add_artifacts(self, 
                      eeg_data: np.ndarray, 
                      artifact_prob: float = 0.3) -> np.ndarray:
        """
        Add realistic artifacts to EEG data (eye blinks, muscle activity, etc.).
        
        Args:
            eeg_data: EEG data matrix of shape (channels, samples)
            artifact_prob: Probability of adding an artifact
            
        Returns:
            EEG data with artifacts
        """
        n_channels, n_samples = eeg_data.shape
        
        # Copy the data
        data_with_artifacts = eeg_data.copy()
        
        # Add eye blinks (mostly in frontal channels)
        if self.rng.random() < artifact_prob:
            # Eye blink timing
            blink_count = self.rng.randint(1, 5)  # 1-4 blinks
            
            for _ in range(blink_count):
                # Random timing for blink
                blink_start = self.rng.randint(0, n_samples - int(0.5 * self.sfreq))
                blink_duration = int(self.rng.uniform(0.2, 0.4) * self.sfreq)  # 200-400ms
                
                # Blink waveform (roughly modeled as a sharp peak)
                blink_shape = signal.gaussian(blink_duration, std=blink_duration/6)
                blink_shape = blink_shape / np.max(blink_shape) * self.rng.uniform(20, 100)
                
                # Apply mostly to frontal channels
                frontal_channels = [i for i, ch in enumerate(self.CHANNEL_NAMES) 
                                  if ch in ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']]
                
                for ch_idx in frontal_channels:
                    # Add with polarity and some random variation
                    ch_modifier = self.rng.uniform(0.7, 1.0)
                    data_with_artifacts[ch_idx, blink_start:blink_start+blink_duration] += blink_shape * ch_modifier
        
        # Add muscle artifacts (high frequency noise bursts)
        if self.rng.random() < artifact_prob:
            # Random channel selection for muscle artifact
            affected_channels = self.rng.choice(n_channels, 
                                              size=self.rng.randint(1, n_channels // 2), 
                                              replace=False)
            
            # Random timing
            artifact_start = self.rng.randint(0, n_samples - int(1.0 * self.sfreq))
            artifact_duration = int(self.rng.uniform(0.3, 1.0) * self.sfreq)  # 300ms to 1s
            
            # Generate high frequency noise
            for ch_idx in affected_channels:
                noise = self.rng.normal(0, self.rng.uniform(5, 20), artifact_duration)
                # Apply a high-pass filter to make it more like EMG
                b, a = signal.butter(4, 20/(self.sfreq/2), 'highpass')
                noise = signal.filtfilt(b, a, noise)
                
                # Add to signal
                end_idx = min(artifact_start + artifact_duration, n_samples)
                noise_to_add = noise[:end_idx - artifact_start]
                data_with_artifacts[ch_idx, artifact_start:end_idx] += noise_to_add
        
        # Add occasional electrode pops (sharp transients)
        if self.rng.random() < artifact_prob * 0.5:  # Less common
            # Random channel
            ch_idx = self.rng.randint(0, n_channels)
            
            # Random timing
            pop_idx = self.rng.randint(0, n_samples)
            
            # Add sharp transient
            pop_amplitude = self.rng.uniform(50, 100)
            pop_sign = 1 if self.rng.random() < 0.5 else -1
            
            # Ensure we don't go out of bounds
            if pop_idx < n_samples:
                data_with_artifacts[ch_idx, pop_idx] += pop_sign * pop_amplitude
        
        return data_with_artifacts
    
    def generate_eeg(self, 
                     mental_state: str, 
                     duration: float, 
                     add_artifacts: bool = True) -> Tuple[np.ndarray, float, List[str]]:
        """
        Generate synthetic EEG data for a specified mental state.
        
        Args:
            mental_state: 'focused' or 'relaxed'
            duration: Length of EEG recording in seconds
            add_artifacts: Whether to add realistic artifacts
            
        Returns:
            Tuple of (EEG data array of shape (channels, samples), 
                     sampling frequency,
                     channel names)
        """
        n_channels = len(self.CHANNEL_NAMES)
        n_samples = int(duration * self.sfreq)
        
        # Create empty EEG array
        eeg_data = np.zeros((n_channels, n_samples))
        
        # Get the profile for this mental state
        if mental_state not in self.MENTAL_STATE_PROFILES:
            raise ValueError(f"Unknown mental state: {mental_state}. "
                           f"Available states: {list(self.MENTAL_STATE_PROFILES.keys())}")
            
        profile = self.MENTAL_STATE_PROFILES[mental_state]
        
        # Generate base signal for each channel
        for ch_idx, channel_name in enumerate(self.CHANNEL_NAMES):
            channel_data = np.zeros(n_samples)
            
            # Base noise (always present)
            noise = self.rng.normal(0, 0.5, n_samples)
            channel_data += noise
            
            # Generate activity in each frequency band
            for band, base_amplitude in profile.items():
                # Apply channel-specific modifiers if available
                ch_modifier = 1.0
                if channel_name in self.CHANNEL_MODIFIERS and band in self.CHANNEL_MODIFIERS[channel_name]:
                    ch_modifier = self.CHANNEL_MODIFIERS[channel_name][band]
                
                # Add some natural variation
                amplitude_variation = self.rng.uniform(0.8, 1.2)
                
                # Final amplitude
                amplitude = base_amplitude * ch_modifier * amplitude_variation
                
                # Generate band signal
                band_signal = self.generate_band_signal(band, duration, amplitude)
                
                # Add to channel data
                channel_data += band_signal
            
            # Add to EEG data
            eeg_data[ch_idx] = channel_data
        
        # Add correlation between channels (spatial coherence)
        eeg_data = self._add_spatial_coherence(eeg_data)
        
        # Add artifacts if requested
        if add_artifacts:
            eeg_data = self.add_artifacts(eeg_data)
        
        return eeg_data, self.sfreq, self.CHANNEL_NAMES
    
    def _add_spatial_coherence(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Add realistic spatial coherence between channels based on proximity.
        
        Args:
            eeg_data: EEG data of shape (channels, samples)
            
        Returns:
            EEG data with added spatial coherence
        """
        n_channels = eeg_data.shape[0]
        
        # Simple approximation - mix nearby channels slightly
        # In a real implementation, this would use actual 3D positions
        coherent_data = eeg_data.copy()
        
        # Define channel neighborhoods (simplified)
        neighborhoods = {
            # Frontal neighborhood
            'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
            # Left temporal
            'left_temporal': ['F7', 'T3', 'T5'],
            # Right temporal
            'right_temporal': ['F8', 'T4', 'T6'],
            # Central
            'central': ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'],
            # Parietal
            'parietal': ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'],
            # Occipital
            'occipital': ['P3', 'Pz', 'P4', 'O1', 'O2'],
        }
        
        # Get indices for each neighborhood
        neighborhood_indices = {}
        for region, channels in neighborhoods.items():
            neighborhood_indices[region] = [i for i, ch in enumerate(self.CHANNEL_NAMES) if ch in channels]
        
        # Mix within neighborhoods
        for region, indices in neighborhood_indices.items():
            if len(indices) > 1:
                # Create a mixture of all channels in this neighborhood
                region_data = eeg_data[indices, :]
                # For each channel, mix in a bit of the neighborhood average
                for i, ch_idx in enumerate(indices):
                    # Exclude the current channel from the average
                    other_indices = [j for j in range(len(indices)) if j != i]
                    if other_indices:  # If there are other channels
                        # Calculate neighborhood average excluding current channel
                        neighborhood_avg = np.mean(region_data[other_indices, :], axis=0)
                        # Mix in neighborhood data (20% neighborhood, 80% original)
                        mix_ratio = self.rng.uniform(0.1, 0.3)  # 10-30% mixing
                        coherent_data[ch_idx, :] = (1 - mix_ratio) * eeg_data[ch_idx, :] + mix_ratio * neighborhood_avg
        
        return coherent_data
    
    def save_to_edf(self, 
                   eeg_data: np.ndarray, 
                   sfreq: float, 
                   ch_names: List[str],
                   filepath: str,
                   subject_info: Dict = None) -> None:
        """
        Save synthetic EEG data to EDF file format using MNE.
        
        Args:
            eeg_data: EEG data array of shape (channels, samples)
            sfreq: Sampling frequency
            ch_names: Channel names
            filepath: Output file path
            subject_info: Optional subject metadata
        """
        # Create MNE info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Add subject info if provided
        if subject_info:
            info['subject_info'] = subject_info
            
        # Create raw object
        raw = mne.io.RawArray(eeg_data, info)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save to EDF format
        raw.export(filepath, fmt='edf', overwrite=True)
        
        logger.info(f"Synthetic EEG data saved to {filepath}")
        
    def generate_dataset(self, 
                         output_dir: str, 
                         n_subjects: int = 30, 
                         conditions: List[str] = None,
                         duration: float = 60.0) -> None:
        """
        Generate a complete dataset of synthetic EEG data for multiple subjects and conditions.
        
        Args:
            output_dir: Directory to save the generated dataset
            n_subjects: Number of subjects to generate
            conditions: List of mental states to generate for each subject
            duration: Duration of each recording in seconds
        """
        if conditions is None:
            conditions = ['focused', 'relaxed']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for subject_idx in range(1, n_subjects + 1):
            subject_id = f"Subject{subject_idx:02d}"
            
            for condition_idx, condition in enumerate(conditions, 1):
                # Generate data
                eeg_data, sfreq, ch_names = self.generate_eeg(
                    mental_state=condition, 
                    duration=duration,
                    add_artifacts=True
                )
                
                # Prepare subject info
                subject_info = {
                    'id': subject_id,
                    'first_name': f"Sub{subject_idx:02d}",
                    'last_name': "Synthetic",
                    'sex': 0,  # 0: unknown, 1: male, 2: female
                    'birthday': (2000, 1, 1)
                }
                
                # Create filename (compatible with our EEGDataLoader)
                filename = f"{subject_id}_{condition_idx}.edf"
                filepath = output_path / filename
                
                # Save to EDF file
                self.save_to_edf(
                    eeg_data=eeg_data,
                    sfreq=sfreq,
                    ch_names=ch_names,
                    filepath=str(filepath),
                    subject_info=subject_info
                )
                
                logger.info(f"Generated {condition} data for {subject_id}")
            
        logger.info(f"Generated synthetic dataset with {n_subjects} subjects "
                  f"and {len(conditions)} conditions in {output_dir}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = SyntheticEEGGenerator(sfreq=500, random_state=42)
    
    # Generate a small test dataset
    output_dir = "data/synthetic_eeg"
    generator.generate_dataset(
        output_dir=output_dir,
        n_subjects=5,
        conditions=['focused', 'relaxed'],
        duration=30.0
    )
    
    logger.info("Synthetic data generation complete!")
