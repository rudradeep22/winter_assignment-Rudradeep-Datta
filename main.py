import os
import subprocess
from pathlib import Path
from datetime import datetime

def check_and_create_data_dir():
    """Create data directory if it doesn't exist"""
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir

def check_data_files():
    """Check if required data files exist"""
    data_dir = check_and_create_data_dir()
    
    # Get most recent data files if they exist
    merchant_files = list(data_dir.glob('merchants_*.csv'))
    transaction_files = list(data_dir.glob('transactions_*.csv'))
    
    # Get latest files if they exist, otherwise use today's date
    if merchant_files and transaction_files:
        merchant_file = max(merchant_files)
        transaction_file = max(transaction_files)
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        merchant_file = data_dir / f'merchants_{date_str}.csv'
        transaction_file = data_dir / f'transactions_{date_str}.csv'
    
    feature_file = data_dir / 'merchant_features_normalized.csv'
    model_file = data_dir / 'autoencoder_model.pth'
    
    return {
        'merchants': merchant_file.exists(),
        'transactions': transaction_file.exists(),
        'merchant_file': merchant_file,
        'transaction_file': transaction_file,
        'features': feature_file.exists(),
        'model': model_file.exists()
    }

def run_pipeline():
    """Run the complete fraud detection pipeline"""
    file_status = check_data_files()
    
    # Generate synthetic data if needed
    if not file_status['merchants'] or not file_status['transactions']:
        print("Generating synthetic merchant and transaction data...")
        subprocess.run(['python', 'data_generation/generate.py'])
        # Recheck files after generation
        file_status = check_data_files()
    
    # Update environment variables for file paths
    os.environ['MERCHANT_FILE'] = str(file_status['merchant_file'])
    os.environ['TRANSACTION_FILE'] = str(file_status['transaction_file'])
    
    # Generate features if needed
    if not file_status['features']:
        print("\nGenerating merchant features...")
        subprocess.run(['python', 'gen_features.py'])
    
    # Train model if needed
    if not file_status['model']:
        print("\nTraining autoencoder model...")
        subprocess.run(['python', 'model.py'])
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
