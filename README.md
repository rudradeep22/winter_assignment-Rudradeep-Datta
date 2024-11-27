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
- Embedded fraud patterns for testing. A single merchant can also exhibit multiple fraudulent behaviours.

The dataset can be tested for fraud and normal transactions by running the `test_generated_data.py` script.  

### Feature Engineering
`gen_features.py` processes transaction data to calculate merchant-level features:
- Transaction velocity metrics (daily/hourly patterns)
- Time-based patterns (night vs business hours)
- Amount distribution statistics
- Customer concentration metrics

All the features are also normalized to prevent scaling issues when training with the autoencoder.

### Pattern Detection
The detection process uses two approaches:
1. An autoencoder (`model.py`) that learns normal merchant behavior and distinguishes between normal and fraudulent behaviour by comparing reconstruction errors.  
2. Pattern-specific scoring (`pattern.py`) that looks for the three fraud patterns by calculating specific scores for them and comparing with the current score.

### Real-time Analysis
The inference engine (`inference.py`) can:
- Analyze transactions in real-time
- Predict if a merchant is fraudulent or not given his transactions 

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

## Further improvements  

This project was created as part of a coding assignment to demonstrate fraud detection techniques using Python and machine learning. In its current state, it cannot detect which fraud pattern is being exhibited by the merchant from autoencoder alone. Also, the inference pipeline doesn't allow uploading a batch of merchants for the model to predict on them.   
Keeping these in mind, some future improvements in this project are:   

* **More rigorous testing of generated data** to ensure the synthetic data looks as real as possible
* **Building autoencoders for each fraud type**, and using ensemble learning to ensure our model is not only able to predict frauds, but also predict its type.  
* **Detailed analysis of merchant features** to ensure which features are important and what other features could improve our model.  
* **Integreating more fraud types** in our pipeline.  
* **Look into better fraud detection techniques** like LSTMs that are better suited for time series data and therefore more capable at looking for fraudulent merchants.