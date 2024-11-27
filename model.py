import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class MerchantDataset(Dataset):
    """Dataset class for merchant transaction features"""
    def __init__(self, features):
        self.features = torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class Autoencoder(nn.Module):
    """Autoencoder model for anomaly detection"""
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Define encoder architecture with gradually decreasing dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Define decoder architecture with gradually increasing dimensions
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """Train the autoencoder model with validation"""
    print("\nStarting model training...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    training_history = {
        'train_losses': [],
        'val_losses': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                val_loss += criterion(reconstructed, batch).item()
        
        # Record losses
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    print("Training completed successfully!")
    return training_history

def calculate_reconstruction_error(model, data_loader):
    """Calculate reconstruction error for anomaly detection"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            reconstructed = model(batch)
            error = torch.mean((batch - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(error.numpy())
    
    return np.array(reconstruction_errors)

def evaluate_model_performance(model, features_df, fraud_merchants, threshold):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    
    # Prepare features for evaluation
    numeric_features = features_df.select_dtypes(include=['float64', 'int64']).astype('float32')
    X = torch.FloatTensor(numeric_features.values)
    
    # Calculate reconstruction errors
    with torch.no_grad():
        reconstructed = model(X)
        errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
    
    # Calculate performance metrics
    fraud_mask = numeric_features.index.isin(fraud_merchants)
    fraud_detection_rate = (errors[fraud_mask] > threshold).mean()
    false_positive_rate = (errors[~fraud_mask] > threshold).mean()
    
    # Print performance summary
    print("\nModel Performance Summary:")
    print(f"Fraud Detection Rate: {fraud_detection_rate:.2%}")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"Average Error (Normal): {errors[~fraud_mask].mean():.6f}")
    print(f"Average Error (Fraud): {errors[fraud_mask].mean():.6f}")
    
    return errors

def plot_error_distribution(errors, fraud_mask, threshold):
    """Visualize error distribution for analysis"""
    plt.figure(figsize=(10, 6))
    plt.hist(errors[~fraud_mask], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(errors[fraud_mask], bins=50, alpha=0.5, label='Fraud', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Detection Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Distribution of Reconstruction Errors')
    plt.savefig('images/error_distribution.png')
    plt.close()

def main():
    """Main execution function"""
    try:
        print("\nStarting model training pipeline...")
        
        # Load and validate transaction data
        transaction_file = os.getenv('TRANSACTION_FILE')
        if not transaction_file:
            raise ValueError("Transaction file path not set in environment")
        
        print("\nLoading and preparing data...")
        # Load and prepare data
        features_df = pd.read_csv('data/merchant_features_normalized.csv')
        if 'merchant_id' in features_df.columns:
            features_df.set_index('merchant_id', inplace=True)
        
        transactions_df = pd.read_csv(transaction_file)
        fraud_merchants = transactions_df[
            transactions_df['velocity_flag'] | 
            transactions_df['time_flag'] | 
            transactions_df['amount_flag']
        ]['merchant_id'].unique()
        
        # Prepare training data (using only normal merchants)
        normal_features = features_df[~features_df.index.isin(fraud_merchants)]
        X = normal_features.select_dtypes(include=['float64', 'int64']).astype('float32').values
        
        # Split data and create data loaders
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        train_dataset = MerchantDataset(X_train)
        val_dataset = MerchantDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        print("\nInitializing and training model...")
        # Initialize and train model
        input_dim = X_train.shape[1]
        model = Autoencoder(input_dim)
        training_history = train_autoencoder(model, train_loader, val_loader, epochs=1000, learning_rate=0.0001)
        
        print("\nEvaluating model performance...")
        # Evaluate model and generate visualizations
        train_errors = calculate_reconstruction_error(model, train_loader)
        threshold = np.percentile(train_errors, 97.5)  # Conservative threshold
        
        all_errors = evaluate_model_performance(model, features_df, fraud_merchants, threshold)
        fraud_mask = features_df.index.isin(fraud_merchants)
        plot_error_distribution(all_errors, fraud_mask, threshold)
        
        print("\nSaving model and metadata...")
        # Save model and training metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': threshold,
            'input_dim': input_dim,
            'training_history': training_history,
            'performance_metrics': {
                'avg_normal_error': float(all_errors[~fraud_mask].mean()),
                'avg_fraud_error': float(all_errors[fraud_mask].mean()),
            }
        }, 'data/autoencoder_model.pth')
        
        print("\nModel pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in model pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
