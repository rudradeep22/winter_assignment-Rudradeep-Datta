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
    def __init__(self, features):
        self.features = torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()  # Normalize outputs between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                val_loss += criterion(output, batch).item()
        
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
    
    return train_losses, val_losses

def calculate_reconstruction_error(model, data_loader):
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch)
            error = torch.mean((batch - output) ** 2, dim=1)
            reconstruction_errors.extend(error.numpy())
    
    return np.array(reconstruction_errors)

def evaluate_model_performance(model, features_df, fraud_merchants, threshold):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    
    # Ensure we're only using numeric columns and converting to float32
    numeric_features = features_df.select_dtypes(include=['float64', 'int64']).astype('float32')
    X = torch.FloatTensor(numeric_features.values)
    
    with torch.no_grad():
        reconstructed = model(X)
        errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
    
    # Calculate metrics
    fraud_mask = numeric_features.index.isin(fraud_merchants)
    fraud_detection = (errors[fraud_mask] > threshold).mean()
    false_positives = (errors[~fraud_mask] > threshold).mean()
    
    print(f"\nModel Performance Metrics:")
    print(f"Fraud Detection Rate: {fraud_detection:.2%}")
    print(f"False Positive Rate: {false_positives:.2%}")
    print(f"Average Error (Normal): {errors[~fraud_mask].mean():.6f}")
    print(f"Average Error (Fraud): {errors[fraud_mask].mean():.6f}")
    
    return errors

def plot_error_distribution(errors, fraud_mask, threshold):
    """Plot error distribution for visual analysis"""
    plt.figure(figsize=(10, 6))
    plt.hist(errors[~fraud_mask], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(errors[fraud_mask], bins=50, alpha=0.5, label='Fraud', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('images/error_distribution.png')
    plt.close()

def main():
    # Load normalized features
    features_df = pd.read_csv('data/merchant_features_normalized.csv')
    
    # Ensure merchant_id is set as index if it's a column
    if 'merchant_id' in features_df.columns:
        features_df.set_index('merchant_id', inplace=True)
    
    # Get transaction file path from environment
    transaction_file = os.getenv('TRANSACTION_FILE', '')
    if not transaction_file:
        raise ValueError("Transaction file path not set in environment")
    
    # Split data into normal and fraudulent merchants
    transactions_df = pd.read_csv(transaction_file)
    fraud_merchants = transactions_df[
        transactions_df['velocity_flag'] | 
        transactions_df['time_flag'] | 
        transactions_df['amount_flag']
    ]['merchant_id'].unique()
    
    # Prepare training data (normal merchants only)
    normal_features = features_df[~features_df.index.isin(fraud_merchants)]
    X = normal_features.select_dtypes(include=['float64', 'int64']).astype('float32').values
    
    # Split into train/validation sets
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = MerchantDataset(X_train)
    val_dataset = MerchantDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    train_losses, val_losses = train_autoencoder(model, train_loader, val_loader, 1000, 0.0001)
    
    # Calculate errors and evaluate
    train_errors = calculate_reconstruction_error(model, train_loader)
    threshold = np.percentile(train_errors, 97.5)  # More conservative threshold
    
    # Evaluate performance
    all_errors = evaluate_model_performance(
        model, features_df, fraud_merchants, threshold
    )
    
    # Plot distributions
    fraud_mask = features_df.index.isin(fraud_merchants)
    plot_error_distribution(all_errors, fraud_mask, threshold)
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'input_dim': input_dim,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'performance_metrics': {
            'avg_normal_error': float(all_errors[~fraud_mask].mean()),
            'avg_fraud_error': float(all_errors[fraud_mask].mean()),
        }
    }, 'data/autoencoder_model.pth')

if __name__ == "__main__":
    main()
