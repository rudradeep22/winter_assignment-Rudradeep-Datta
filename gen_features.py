import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, Any

class MerchantFeatureGenerator:
    """Generate and process merchant features from transaction data"""
    
    def __init__(self, output_dir: str = 'data'):
        """Initialize feature generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nInitialized feature generator with output directory: {output_dir}")
    
    def calculate_velocity_metrics(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction velocity-related features"""
        print("\nCalculating velocity metrics...")
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Calculate daily transaction counts
        daily_txns = df.groupby(['merchant_id', 'date']).size().reset_index()
        daily_txns.columns = ['merchant_id', 'date', 'daily_count']
        
        # Calculate hourly transaction counts
        hourly_txns = df.groupby(['merchant_id', 'date', 'hour']).size().reset_index()
        hourly_txns.columns = ['merchant_id', 'date', 'hour', 'hourly_count']
        
        # Calculate daily statistics
        daily_stats = daily_txns.groupby('merchant_id')['daily_count'].agg([
            ('daily_avg_txns', 'mean'),
            ('daily_std_txns', 'std')
        ]).fillna(0)
        
        # Calculate hourly statistics
        hourly_stats = hourly_txns.groupby('merchant_id')['hourly_count'].agg([
            ('hourly_avg_txns', 'mean'),
            ('max_txns_hour', 'max')
        ])
        
        # Combine and calculate velocity ratio
        velocity_features = daily_stats.join(hourly_stats)
        velocity_features['velocity_ratio'] = (
            velocity_features['max_txns_hour'] / 
            velocity_features['hourly_avg_txns']
        ).fillna(0)
        
        print(f"Calculated velocity metrics for {len(velocity_features)} merchants")
        return velocity_features
    
    def calculate_time_patterns(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based pattern features"""
        print("\nCalculating time-based patterns...")
        df = transactions_df.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        time_features = []
        
        merchant_count = len(df['merchant_id'].unique())
        for i, merchant_id in enumerate(df['merchant_id'].unique(), 1):
            if i % 100 == 0:
                print(f"Processing merchant {i}/{merchant_count}")
                
            merchant_txns = df[df['merchant_id'] == merchant_id]
            total_txns = len(merchant_txns)
            
            features = {
                'merchant_id': merchant_id,
                'night_txn_ratio': len(merchant_txns[
                    merchant_txns['hour'].isin([23, 0, 1, 2, 3, 4])
                ]) / total_txns,
                'business_hour_ratio': len(merchant_txns[
                    merchant_txns['hour'].between(9, 18)
                ]) / total_txns,
                'evening_ratio': len(merchant_txns[
                    merchant_txns['hour'].between(19, 22)
                ]) / total_txns
            }
            time_features.append(features)
        
        result = pd.DataFrame(time_features).set_index('merchant_id')
        print(f"Calculated time patterns for {len(result)} merchants")
        return result
    
    def calculate_amount_patterns(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction amount-related features"""
        return transactions_df.groupby('merchant_id')['amount'].agg([
            ('avg_amount', 'mean'),
            ('std_amount', 'std'),
            ('min_amount', 'min'),
            ('max_amount', 'max'),
            ('amount_skew', 'skew')
        ]).fillna(0)
    
    def calculate_customer_patterns(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer concentration features"""
        customer_features = []
        
        for merchant_id in transactions_df['merchant_id'].unique():
            merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id]
            customer_counts = merchant_txns['customer_id'].value_counts()
            
            features = {
                'merchant_id': merchant_id,
                'unique_customers': len(customer_counts),
                'top_customer_ratio': customer_counts.iloc[0] / len(merchant_txns),
                'top_5_customer_ratio': customer_counts.iloc[:5].sum() / len(merchant_txns)
            }
            customer_features.append(features)
        
        return pd.DataFrame(customer_features).set_index('merchant_id')
    
    def plot_feature_distributions(self, features_df: pd.DataFrame):
        """Generate and save feature distribution plots"""
        n_features = len(features_df.columns)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, column in enumerate(features_df.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(features_df[column], kde=True)
            plt.title(column)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("images/feature_distributions.png")
        plt.close()
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        scaler = StandardScaler()
        normalized_df = pd.DataFrame(
            scaler.fit_transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
        return normalized_df
    
    def generate_all_features(self, transactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate all feature sets from transaction data"""
        print("\nGenerating all features...")
        
        # Calculate all feature groups
        velocity_features = self.calculate_velocity_metrics(transactions_df)
        time_features = self.calculate_time_patterns(transactions_df)
        amount_features = self.calculate_amount_patterns(transactions_df)
        customer_features = self.calculate_customer_patterns(transactions_df)
        
        # Combine all features
        print("\nCombining feature groups...")
        all_features = pd.concat([
            velocity_features,
            time_features,
            amount_features,
            customer_features
        ], axis=1)
        
        # Generate normalized features
        print("\nNormalizing features...")
        normalized_features = self.normalize_features(all_features)
        
        print(f"\nCompleted feature generation for {len(normalized_features)} merchants")
        return {
            'raw_features': all_features,
            'normalized_features': normalized_features
        }

def main():
    """Main execution function"""
    print("\nStarting feature generation process...")
    
    # Get transaction file path
    transaction_file = os.getenv('TRANSACTION_FILE', '')
    if not transaction_file:
        raise ValueError("Transaction file path not set in environment")
    
    # Load transaction data
    print(f"\nLoading transaction data from: {transaction_file}")
    transactions_df = pd.read_csv(transaction_file)
    print(f"Loaded {len(transactions_df)} transactions")
    
    # Initialize feature generator and generate features
    generator = MerchantFeatureGenerator()
    features = generator.generate_all_features(transactions_df)
    
    # Save normalized features
    print("\nSaving normalized features...")
    features['normalized_features'].to_csv('data/merchant_features_normalized.csv')
    
    # Generate and save feature distribution plots
    print("\nGenerating feature distribution plots...")
    generator.plot_feature_distributions(features['normalized_features'])
    
    # Print summary statistics
    print("\nFeature Generation Summary:")
    print(f"Total merchants processed: {len(features['normalized_features'])}")
    print("\nFeature Statistics:")
    print(features['normalized_features'].describe())
    
    print("\nFeature generation completed successfully!")

if __name__ == "__main__":
    main()