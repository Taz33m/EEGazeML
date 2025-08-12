"""
Preprocessing module for EEG mental state classifier.
Handles filtering, artifact removal, and data cleaning.
"""

import numpy as np
import mne
from typing import Tuple, Dict, List, Optional


class EEGPreprocessor:
    """Preprocesses raw EEG data to clean signals and remove artifacts."""
    
    def __init__(self, 
                 l_freq: float = 0.5, 
                 h_freq: float = 45.0,
                 notch_freq: float = 60.0,
                 ica_components: int = 15):
        """
        Initialize the preprocessor.
        
        Args:
            l_freq: Lower frequency bound for bandpass filter (Hz)
            h_freq: Upper frequency bound for bandpass filter (Hz)
            notch_freq: Frequency for notch filter (Hz), typically power line frequency
            ica_components: Number of ICA components for artifact removal
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.ica_components = ica_components
    
    def preprocess(self, raw: mne.io.Raw, 
                   apply_notch: bool = True,
                   remove_artifacts: bool = True) -> mne.io.Raw:
        """
        Apply preprocessing pipeline to raw EEG data.
        
        Args:
            raw: MNE Raw object containing EEG data
            apply_notch: Whether to apply notch filter for line noise
            remove_artifacts: Whether to perform artifact removal with ICA
            
        Returns:
            Preprocessed MNE Raw object
        """
        # Create a copy of the raw data to avoid modifying the original
        raw_processed = raw.copy()
        
        # Apply bandpass filter
        raw_processed.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                            picks='eeg', method='fir', phase='zero')
        
        # Apply notch filter to remove power line noise if requested
        if apply_notch:
            raw_processed.notch_filter(freqs=self.notch_freq, picks='eeg')
        
        # Remove artifacts using ICA if requested
        if remove_artifacts:
            raw_processed = self._remove_artifacts(raw_processed)
        
        return raw_processed
    
    def _remove_artifacts(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Remove artifacts from EEG data using ICA.
        
        Args:
            raw: MNE Raw object containing EEG data
            
        Returns:
            MNE Raw object with artifacts removed
        """
        # Create a copy of the raw data
        raw_clean = raw.copy()
        
        # Set up and fit ICA
        ica = mne.preprocessing.ICA(
            n_components=self.ica_components,
            random_state=42,
            method='fastica'
        )
        
        # Fit ICA on the data
        ica.fit(raw_clean)
        
        # Find EOG artifacts (eye blinks and movements)
        # This requires EOG channels - if not available, we can use correlation with frontal EEG channels
        eog_indices = []
        
        # Check if EOG channels exist
        if 'eog' in raw_clean:
            # Find components that correlate with EOG
            eog_indices, _ = ica.find_bads_eog(raw_clean)
        else:
            # Try to find blink artifacts based on typical patterns
            # This is a heuristic approach and may need tuning
            eog_indices = self._find_blink_components(ica, raw_clean)
        
        # Find and exclude components that are likely to be artifacts
        # Mark these components for exclusion
        ica.exclude = eog_indices
        
        # Apply the ICA to remove the marked components
        ica.apply(raw_clean)
        
        return raw_clean
    
    def _find_blink_components(self, ica: mne.preprocessing.ICA, 
                              raw: mne.io.Raw) -> List[int]:
        """
        Find ICA components that likely correspond to eye blinks.
        This is a heuristic approach when EOG channels are not available.
        
        Args:
            ica: Fitted ICA object
            raw: MNE Raw object containing EEG data
            
        Returns:
            List of indices of components likely to be eye blinks
        """
        # This is a placeholder for a more sophisticated approach
        # In a real implementation, we would look at properties of the components
        # such as their topographies and time courses
        
        # For now, just return an empty list as a placeholder
        # In a real implementation, you might use a classifier or heuristics
        # to identify blink components
        return []
    
    def epoch_data(self, raw: mne.io.Raw, 
                   tmin: float = -0.2, 
                   tmax: float = 0.5,
                   event_id: Dict = None) -> mne.Epochs:
        """
        Create epochs from continuous data.
        
        Args:
            raw: MNE Raw object containing EEG data
            tmin: Start time of epoch relative to event (in seconds)
            tmax: End time of epoch relative to event (in seconds)
            event_id: Dictionary mapping event descriptions to integer IDs
            
        Returns:
            MNE Epochs object
        """
        # Find events in the data
        # This is a placeholder - in reality, events would be extracted from 
        # markers in the data or from a separate events file
        events = mne.find_events(raw)
        
        # If no event_id is provided, create a simple one
        if event_id is None:
            event_id = {'unknown': 1}
        
        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=event_id, 
                           tmin=tmin, tmax=tmax, 
                           baseline=(tmin, 0), preload=True)
        
        return epochs
