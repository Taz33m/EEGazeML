"""
Data loading module for EEG mental state classifier.
Handles loading raw EEG data from the PhysioNet EEGMat dataset.
"""

import os
import numpy as np
import pandas as pd
import mne
from typing import Tuple, Dict, List, Optional


class EEGDataLoader:
    """Loads and provides access to the EEG dataset."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing the EEG dataset
        """
        self.data_dir = data_dir
        self.subjects = []
        self.file_paths = {}
        self._discover_data_files()
    
    def _discover_data_files(self):
        """Scan the data directory and identify available data files."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Based on actual PhysioNet EEGMat dataset structure
        # Files are named like: Subject01_1.edf and Subject01_2.edf
        # where 1 typically corresponds to arithmetic task and 2 to baseline
        files = os.listdir(self.data_dir)
        subject_ids = set()
        
        for file in files:
            if file.endswith('.edf') or file.endswith('.mat'):
                # Check if file matches pattern Subject##_#.edf
                if file.startswith('Subject') and '_' in file:
                    # Extract the subject number
                    subject_part = file.split('_')[0]
                    subject_id = subject_part.replace('Subject', '')
                    
                    # Extract the condition number
                    condition_part = file.split('_')[1].split('.')[0]
                    
                    # Add to subjects set
                    subject_ids.add(subject_id)
                    
                    # Initialize file paths dict for this subject if not exists
                    if subject_id not in self.file_paths:
                        self.file_paths[subject_id] = {}
                    
                    # Map condition numbers to condition names
                    # For PhysioNet data: 1 = arithmetic (mental task), 2 = baseline (relaxed)
                    # For synthetic data: 1 = focused, 2 = relaxed
                    condition_map = {
                        '1': ['arithmetic', 'focused'],  # Add both possible names for condition 1
                        '2': ['baseline', 'relaxed']     # Add both possible names for condition 2
                    }
                    
                    # Store paths for all possible condition names
                    for cond_name in condition_map.get(condition_part, []):
                        self.file_paths[subject_id][cond_name] = os.path.join(self.data_dir, file)
        
        self.subjects = sorted(list(subject_ids))
        print(f"Found data for {len(self.subjects)} subjects")
    
    def load_raw_data(self, subject_id: str, condition: str) -> Optional[mne.io.Raw]:
        """
        Load raw EEG data for a specific subject and condition.
        
        Args:
            subject_id: Subject identifier
            condition: 'arithmetic' or 'baseline'
            
        Returns:
            MNE Raw object containing the EEG data
        """
        if subject_id not in self.file_paths:
            raise ValueError(f"No data found for subject {subject_id}")
        
        if condition not in self.file_paths[subject_id]:
            raise ValueError(f"No {condition} data found for subject {subject_id}")
        
        if condition not in self.file_paths[subject_id]:
            return None
        
        file_path = self.file_paths[subject_id][condition]
        
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_epochs(self, subject_id: str, tmin=-0.2, tmax=0.5, baseline=(None, 0)):
        """
        Create epochs from raw data for both conditions.
        
        Args:
            subject_id: Subject identifier
            tmin: Start time of epoch in seconds
            tmax: End time of epoch in seconds
            baseline: Baseline period for baseline correction
            
        Returns:
            Dictionary of MNE Epochs objects for each condition
        """
        epochs_dict = {}
        
        for condition in ['arithmetic', 'baseline']:
            raw = self.load_raw_data(subject_id, condition)
            
            if raw is None:
                continue
            
            # Create events - for continuous data we'll create equidistant events
            # Sample rate is raw.info['sfreq']
            # Create an event every 1 second
            sample_rate = int(raw.info['sfreq'])
            event_spacing = sample_rate  # 1 event per second
            n_samples = len(raw.times)
            
            # Create events at regular intervals
            events = np.array([
                [i, 0, 1] for i in range(0, n_samples, event_spacing)
                if i + int(tmax * sample_rate) < n_samples
            ])
            
            if len(events) > 0:
                # Create epochs
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, 
                                    baseline=baseline, preload=True, verbose=False)
                epochs_dict[condition] = epochs
        
        return epochs_dict
    
    def _convert_mat_to_mne(self, mat_data: Dict) -> mne.io.Raw:
        """
        Convert MATLAB data to MNE Raw format.
        Specific implementation depends on the structure of the .mat files.
        
        Args:
            mat_data: Dictionary containing data loaded from .mat file
            
        Returns:
            MNE Raw object
        """
        # This is a placeholder - implementation will depend on actual data format
        # Example implementation (will need to be modified based on actual data):
        
        # Assume mat_data contains:
        # - 'data': EEG signal array (channels × time)
        # - 'sfreq': sampling frequency
        # - 'ch_names': channel names
        
        if 'data' not in mat_data:
            raise ValueError("Expected 'data' field in .mat file")
            
        data = mat_data.get('data')
        sfreq = mat_data.get('sfreq', 250)  # Default to 250 Hz if not specified
        
        # If channel names are provided, use them, otherwise create generic names
        if 'ch_names' in mat_data:
            ch_names = mat_data['ch_names']
        else:
            ch_names = [f'EEG{i+1}' for i in range(data.shape[0])]
        
        # Create info structure
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create raw object
        raw = mne.io.RawArray(data, info)
        return raw
    
    def get_subject_list(self) -> List[str]:
        """
        Get a list of all available subject IDs.
        
        Returns:
            List of subject IDs
        """
        return self.subjects
    
    def get_available_conditions(self, subject_id: str) -> List[str]:
        """
        Get available conditions for a specific subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            List of available conditions ('arithmetic', 'baseline')
        """
        if subject_id not in self.file_paths:
            return []
        
        return list(self.file_paths[subject_id].keys())
