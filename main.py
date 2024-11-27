import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class FraudDetectionPipeline:
    """Main pipeline for merchant fraud detection system"""
    
    def __init__(self):
        """Initialize the pipeline"""
        self.data_dir = self.check_and_create_data_dir()
    
    def check_and_create_data_dir(self) -> Path:
        """Create and verify data directory
        
        Returns:
            Path object pointing to data directory
        """
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    def check_data_files(self) -> Dict[str, bool]:
        """Check existence of required data files
        
        Returns:
            Dictionary containing file status and paths
        """
        # Get most recent data files if they exist
        merchant_files = list(self.data_dir.glob('merchants_*.csv'))
        transaction_files = list(self.data_dir.glob('transactions_*.csv'))
        
        # Determine file paths
        if merchant_files and transaction_files:
            merchant_file = max(merchant_files)
            transaction_file = max(transaction_files)
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            merchant_file = self.data_dir / f'merchants_{date_str}.csv'
            transaction_file = self.data_dir / f'transactions_{date_str}.csv'
        
        feature_file = self.data_dir / 'merchant_features_normalized.csv'
        model_file = self.data_dir / 'autoencoder_model.pth'
        
        return {
            'merchants': merchant_file.exists(),
            'transactions': transaction_file.exists(),
            'merchant_file': merchant_file,
            'transaction_file': transaction_file,
            'features': feature_file.exists(),
            'model': model_file.exists()
        }
    
    def run_script(self, script_path: str, description: str):
        """Run a Python script and log its execution
        
        Args:
            script_path: Path to the Python script
            description: Description of the script's purpose
        """
        try:
            print(f"\nStarting {description}...")
            result = subprocess.run(['python', script_path], check=True)
            if result.returncode == 0:
                print(f"Successfully completed {description}")
            else:
                print(f"Failed to complete {description}")
                raise subprocess.CalledProcessError(result.returncode, script_path)
        except Exception as e:
            print(f"Error in {description}: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Execute the complete fraud detection pipeline"""
        try:
            print("\nStarting fraud detection pipeline...")
            
            # Check required files
            file_status = self.check_data_files()
            
            # Generate synthetic data if needed
            if not file_status['merchants'] or not file_status['transactions']:
                print("\nGenerating synthetic data...")
                self.run_script('data_generation/generate.py', "data generation")
                # Recheck files after generation
                file_status = self.check_data_files()
            
            # Set environment variables for file paths
            os.environ['MERCHANT_FILE'] = str(file_status['merchant_file'])
            os.environ['TRANSACTION_FILE'] = str(file_status['transaction_file'])
            
            # Generate features if needed
            if not file_status['features']:
                print("\nGenerating merchant features...")
                self.run_script('gen_features.py', "feature generation")
            
            # Train model if needed
            if not file_status['model']:
                print("\nTraining autoencoder model...")
                self.run_script('model.py', "model training")
            
            # Run pattern detection
            print("\nRunning pattern detection...")
            self.run_script('pattern.py', "pattern detection")
            
            print("\nFraud detection pipeline completed successfully!")
            
        except Exception as e:
            print(f"\nError: Pipeline failed - {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        pipeline = FraudDetectionPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
