# Merchant Fraud Detection System

## Overview
This is a Python-based system that detects fraudulent merchant behavior using machine learning. It uses an autoencoder for anomaly detection combined with specific pattern recognition to identify three main types of fraud:

- High-volume transaction spikes
- Suspicious late-night activity
- Customer concentration (where few customers make most transactions)

## How It Works

### Data Generation
The system starts by generating synthetic data (in `data_generation/generate.py`) that includes:
- Merchant profiles with basic business information
- Transaction records with timestamps, amounts, and customer details
- Embedded fraud patterns for testing

### Feature Engineering
`gen_features.py` processes transaction data to calculate merchant-level features:
- Transaction velocity metrics (daily/hourly patterns)
- Time-based patterns (night vs business hours)
- Amount distribution statistics
- Customer concentration metrics

### Pattern Detection
The detection process uses two approaches:
1. An autoencoder (`model.py`) that learns normal merchant behavior
2. Pattern-specific scoring (`pattern.py`) that looks for known fraud patterns

### Real-time Analysis
The inference engine (`inference.py`) can:
- Analyze transactions in real-time
- Process single merchants

## Project Structure
```
merchant-fraud-detection/
├── data/                      # Data storage
├── data_generation/          
│   ├── generate.py           # Creates synthetic data
│   └── test_generated_data.py # Tests data generation
├── images/                   # Visualization outputs
├── gen_features.py          # Feature engineering
├── model.py                 # Autoencoder model
├── pattern.py              # Pattern detection
├── inference.py            # Real-time analysis
└── main.py                 # Main pipeline
```

## How to Run

1. Set up the environment:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python main.py
```

Or run individual components:

```bash
# Generate test data
python data_generation/generate.py

# Process features
python gen_features.py

# Train model
python model.py

# Detect patterns
python pattern.py

# Run real-time analysis
python inference.py
```

## Output Files
- `data/merchants_*.csv`: Merchant profiles
- `data/transactions_*.csv`: Transaction records
- `data/merchant_features_normalized.csv`: Processed features
- `data/pattern_detection_results.csv`: Detection results
- `images/`: Visualization plots

## Testing
Run the tests with:
```bash
pytest data_generation/test_generated_data.py
```

---

This project was created as part of a coding assignment to demonstrate fraud detection techniques using Python and machine learning.