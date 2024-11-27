import torch
import pandas as pd
from model import Autoencoder

class FraudDetector:
    def __init__(self, model_path: str = 'data/autoencoder_model.pth'):
        """Initialize with trained model"""
        self.model_data = torch.load(model_path)
        self.model = self._load_model()
        self.threshold = self.model_data['threshold']
        self.model.eval()
        
    def _load_model(self):
        """Load the trained autoencoder model"""
        model = Autoencoder(self.model_data['input_dim'])
        model.load_state_dict(self.model_data['model_state_dict'])
        return model
    
    def predict(self, merchant_id: str) -> dict:
        """Predict if a merchant is fraudulent based on reconstruction error"""
        try:
            # Load normalized features
            features_df = pd.read_csv('data/merchant_features_normalized.csv')
            features_df.set_index('merchant_id', inplace=True)
            
            # Check if merchant exists
            if merchant_id not in features_df.index:
                raise ValueError(f"Merchant {merchant_id} not found in features dataset")
            
            # Get merchant features and convert to tensor
            merchant_features = features_df.loc[merchant_id]
            X = torch.FloatTensor(merchant_features.values.reshape(1, -1))
            
            # Get reconstruction error
            with torch.no_grad():
                reconstructed = self.model(X)
                reconstruction_error = torch.mean((X - reconstructed) ** 2).item()
            
            # Determine if fraudulent
            is_fraudulent = reconstruction_error > self.threshold
            
            return {
                'merchant_id': merchant_id,
                'is_fraudulent': is_fraudulent,
                'reconstruction_error': reconstruction_error,
                'threshold': self.threshold
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

def main():
    try:
        # Initialize detector
        detector = FraudDetector()
        
        # Get merchant ID from user
        merchant_id = input("Enter merchant ID to analyze: ")
        
        # Run prediction
        result = detector.predict(merchant_id)
        
        # Print results
        print("\nFraud Detection Results:")
        print(f"Merchant ID: {result['merchant_id']}")
        print(f"Is Fraudulent: {result['is_fraudulent']}")
        print(f"Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"Threshold: {result['threshold']:.6f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
