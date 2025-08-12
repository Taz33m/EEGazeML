# EEG Mental State Classifier

This application analyzes EEG data to classify mental states, specifically distinguishing between focused states (mental arithmetic) and relaxed baseline states.

This is an older project with tweaks that have been made to make it more open-source friendly. Any and all PRs welcome.

## Project Overview

This system:
- Reads raw EEG signals from the PhysioNet EEGMat dataset
- Filters and cleans data to remove noise/artifacts
- Extracts brainwave features (alpha, beta, etc.) linked to mental states
- Uses machine learning to classify mental states
- Provides an API for instant prediction from uploaded EEG data
- Displays results in a real-time web dashboard

## Project Structure

```
eeg_mental_state_classifier/
├── data/                    # Dataset storage
├── notebooks/               # Jupyter notebooks for exploration and visualization
├── src/                     # Source code
│   ├── data_processing/     # Data loading and preprocessing
│   ├── feature_extraction/  # Brainwave feature extraction
│   ├── models/              # ML models for state classification
│   ├── api/                 # API for serving predictions
│   └── dashboard/           # Web dashboard
├── tests/                   # Unit and integration tests
└── config/                  # Configuration files
```

## Setup and Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the dataset and place it in the `data/` directory

## Usage

### Data Processing
```
python src/data_processing/process_data.py
```

### Model Training
```
python src/models/train_model.py
```

### Running the API
```
python src/api/app.py
```

### Starting the Dashboard
```
python src/dashboard/app.py
```

## License
[MIT License](LICENSE)
