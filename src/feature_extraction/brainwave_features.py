"""
Feature extraction module for EEG mental state classifier.
Extracts brainwave features linked to mental states from EEG data.
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from typing import Dict, List, Tuple, Union
import pywt


class BrainwaveFeatureExtractor:
    """Extracts various brainwave features from preprocessed EEG data."""
    
    # Define frequency bands of interest
    FREQ_BANDS = {
        'delta': (0.5, 4),    # Sleep, deep relaxation
        'theta': (4, 8),      # Drowsiness, meditation, creativity
        'alpha': (8, 13),     # Relaxation, calmness, reflection
        'beta': (13, 30),     # Active thinking, focus, alertness
        'gamma': (30, 45)     # Higher cognitive processing, perception
    }
    
    def __init__(self, sfreq: float = 250.0):
        """
        Initialize the feature extractor.
        
        Args:
            sfreq: Sampling frequency of the EEG data (Hz)
        """
        self.sfreq = sfreq
    
    def extract_features(self, 
                         data: Union[mne.io.Raw, mne.Epochs, np.ndarray],
                         include_bands: List[str] = None,
                         include_connectivity: bool = False,
                         include_complexity: bool = False) -> pd.DataFrame:
        """
        Extract features from EEG data.
        
        Args:
            data: EEG data (MNE Raw/Epochs object or numpy array)
            include_bands: List of frequency bands to include 
                           (default: all bands)
            include_connectivity: Whether to include connectivity features
            include_complexity: Whether to include complexity measures
            
        Returns:
            DataFrame with extracted features
        """
        # Determine which bands to extract
        if include_bands is None:
            include_bands = list(self.FREQ_BANDS.keys())
        
        # Convert MNE object to numpy array if needed
        # Handle any class derived from mne.io.Raw
        if hasattr(data, 'get_data') and hasattr(data, 'ch_names'):
            # Get numpy array and channel names from Raw-like object
            signals = data.get_data()
            ch_names = data.ch_names
        elif isinstance(data, mne.Epochs):
            signals = data.get_data()
            # For epochs, average across epochs
            signals = np.mean(signals, axis=0)
            ch_names = data.ch_names
        elif isinstance(data, np.ndarray):
            # Assume numpy array is provided with shape (channels, samples)
            signals = data
            # Generate generic channel names
            ch_names = [f'Ch{i+1}' for i in range(signals.shape[0])]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected mne.io.Raw, mne.Epochs, or numpy.ndarray")
        
        # Initialize features dictionary
        features = {}
        
        # Extract band power features
        band_powers = self._extract_band_powers(signals, include_bands)
        features.update(band_powers)
        
        # Extract band power ratios
        power_ratios = self._extract_power_ratios(band_powers)
        features.update(power_ratios)
        
        # Extract connectivity features if requested
        if include_connectivity:
            connectivity = self._extract_connectivity(signals, include_bands)
            features.update(connectivity)
        
        # Extract complexity measures if requested
        if include_complexity:
            complexity = self._extract_complexity_measures(signals)
            features.update(complexity)
        
        # Convert to DataFrame - for scalar values, we need to create a DataFrame with a single row
        df = pd.DataFrame(features, index=[0])
        
        return df
    
    def _extract_band_powers(self, signals: np.ndarray, 
                            include_bands: List[str]) -> Dict:
        """
        Extract power in different frequency bands.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            include_bands: List of frequency bands to include
            
        Returns:
            Dictionary with band power features
        """
        features = {}
        n_channels = signals.shape[0]
        
        for band_name in include_bands:
            if band_name not in self.FREQ_BANDS:
                continue
                
            fmin, fmax = self.FREQ_BANDS[band_name]
            
            # Calculate power spectral density
            for ch_idx in range(n_channels):
                # Compute power spectrum
                f, psd = signal.welch(signals[ch_idx], fs=self.sfreq, 
                                     nperseg=min(256, signals.shape[1]//2))
                
                # Find indices corresponding to the frequency band
                idx_band = np.logical_and(f >= fmin, f <= fmax)
                
                # Calculate average power in the band
                power = np.mean(psd[idx_band])
                
                # Store in features dictionary
                feature_name = f"{band_name}_power_ch{ch_idx+1}"
                features[feature_name] = power
                
            # Also compute average across channels
            features[f"{band_name}_power_avg"] = np.mean(
                [features[f"{band_name}_power_ch{ch_idx+1}"] for ch_idx in range(n_channels)]
            )
        
        return features
    
    def _extract_power_ratios(self, band_powers: Dict) -> Dict:
        """
        Calculate ratios between different frequency bands.
        
        Args:
            band_powers: Dictionary with band power features
            
        Returns:
            Dictionary with band power ratio features
        """
        features = {}
        
        # Define interesting ratios
        ratios = [
            ('theta', 'beta'),    # Relaxation vs. focus
            ('alpha', 'beta'),    # Relaxation vs. active thinking
            ('theta', 'alpha'),   # Meditation vs. relaxed alertness
            ('delta', 'beta')     # Deep relaxation vs. active thinking
        ]
        
        # Calculate average band powers across channels
        for numerator, denominator in ratios:
            num_key = f"{numerator}_power_avg"
            denom_key = f"{denominator}_power_avg"
            
            if num_key in band_powers and denom_key in band_powers:
                ratio_name = f"{numerator}_{denominator}_ratio"
                
                # Avoid division by zero
                if band_powers[denom_key] > 0:
                    ratio = band_powers[num_key] / band_powers[denom_key]
                else:
                    ratio = 0
                    
                features[ratio_name] = ratio
        
        return features
    
    def _extract_connectivity(self, signals: np.ndarray, 
                             include_bands: List[str]) -> Dict:
        """
        Extract connectivity features between channels.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            include_bands: List of frequency bands to include
            
        Returns:
            Dictionary with connectivity features
        """
        features = {}
        n_channels = signals.shape[0]
        
        # Calculate band-filtered signals
        filtered_signals = {}
        for band_name in include_bands:
            if band_name not in self.FREQ_BANDS:
                continue
                
            fmin, fmax = self.FREQ_BANDS[band_name]
            band_filtered = np.zeros_like(signals)
            
            for ch_idx in range(n_channels):
                # Apply bandpass filter
                b, a = signal.butter(4, [fmin, fmax], btype='bandpass', fs=self.sfreq)
                band_filtered[ch_idx] = signal.filtfilt(b, a, signals[ch_idx])
            
            filtered_signals[band_name] = band_filtered
        
        # Calculate correlation-based connectivity
        for band_name, band_filtered in filtered_signals.items():
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(band_filtered)
            
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n_channels, k=1)
            correlations = corr_matrix[triu_indices]
            
            # Average correlation as a global connectivity measure
            features[f"{band_name}_connectivity_avg"] = np.mean(correlations)
            
            # Also include max correlation as a feature
            features[f"{band_name}_connectivity_max"] = np.max(correlations) if len(correlations) > 0 else 0
        
        return features
    
    def _extract_complexity_measures(self, signals: np.ndarray) -> Dict:
        """
        Extract complexity measures from EEG signals.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            
        Returns:
            Dictionary with complexity features
        """
        features = {}
        n_channels = signals.shape[0]
        
        for ch_idx in range(n_channels):
            # Calculate sample entropy (approximate)
            # Note: A full implementation would use a dedicated library
            # This is a simplified version
            signal = signals[ch_idx]
            features[f"sample_entropy_ch{ch_idx+1}"] = self._approximate_entropy(signal)
            
            # Calculate Higuchi fractal dimension
            features[f"fractal_dim_ch{ch_idx+1}"] = self._higuchi_fd(signal)
        
        # Average across channels
        features["sample_entropy_avg"] = np.mean([
            features[f"sample_entropy_ch{ch_idx+1}"] for ch_idx in range(n_channels)
        ])
        
        features["fractal_dim_avg"] = np.mean([
            features[f"fractal_dim_ch{ch_idx+1}"] for ch_idx in range(n_channels)
        ])
        
        return features
    
    def _approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate approximate entropy of a signal.
        
        Args:
            signal: 1D signal
            m: Pattern length
            r: Similarity threshold
            
        Returns:
            Approximate entropy value
        """
        # This is a placeholder for a proper entropy calculation
        # A real implementation would use a library like EntropyHub or pyEntropy
        # For now, return standard deviation as a crude approximation
        return np.std(signal)
    
    def _higuchi_fd(self, signal: np.ndarray, kmax: int = 10) -> float:
        """
        Calculate Higuchi fractal dimension of a signal.
        
        Args:
            signal: 1D signal
            kmax: Maximum lag
            
        Returns:
            Higuchi fractal dimension
        """
        # This is a placeholder for a proper fractal dimension calculation
        # A real implementation would compute the actual Higuchi algorithm
        # For now, return a simple measure of signal variability
        return np.var(np.diff(signal)) / np.var(signal)
    
    def extract_wavelet_features(self, signals: np.ndarray, 
                                wavelet: str = 'db4',
                                level: int = 5) -> Dict:
        """
        Extract wavelet-based features from EEG signals.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary with wavelet features
        """
        features = {}
        n_channels = signals.shape[0]
        
        for ch_idx in range(n_channels):
            signal = signals[ch_idx]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # Extract features from each coefficient level
            for i, coef in enumerate(coeffs):
                # Calculate energy
                energy = np.sum(coef**2) / len(coef)
                features[f"wavelet_energy_ch{ch_idx+1}_level{i}"] = energy
                
                # Calculate entropy
                if np.any(coef):
                    p = coef**2 / np.sum(coef**2)
                    p = p[p > 0]  # Avoid log(0)
                    entropy = -np.sum(p * np.log2(p))
                else:
                    entropy = 0
                features[f"wavelet_entropy_ch{ch_idx+1}_level{i}"] = entropy
        
        return features
