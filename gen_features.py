import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os

def calculate_velocity_metrics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate transaction velocity metrics"""
    df = transactions_df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    daily_txns = df.groupby(['merchant_id', 'date']).size().reset_index()
    daily_txns.columns = ['merchant_id', 'date', 'daily_count']
    
    hourly_txns = df.groupby(['merchant_id', 'date', 'hour']).size().reset_index()
    hourly_txns.columns = ['merchant_id', 'date', 'hour', 'hourly_count']
    
    daily_stats = daily_txns.groupby('merchant_id')['daily_count'].agg([
        ('daily_avg_txns', 'mean'),
        ('daily_std_txns', 'std')
    ]).fillna(0)
    
    hourly_stats = hourly_txns.groupby('merchant_id')['hourly_count'].agg([
        ('hourly_avg_txns', 'mean'),
        ('max_txns_hour', 'max')
    ])
    
    velocity_features = daily_stats.join(hourly_stats)
    velocity_features['velocity_ratio'] = (
        velocity_features['max_txns_hour'] / 
        velocity_features['hourly_avg_txns']
    ).fillna(0)
    
    return velocity_features

def calculate_time_patterns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based pattern metrics"""
    df = transactions_df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    time_features = pd.DataFrame()
    
    # Calculate percentage of transactions in different time periods
    for merchant_id in df['merchant_id'].unique():
        merchant_txns = df[df['merchant_id'] == merchant_id]
        total_txns = len(merchant_txns)
        
        features = {
            'merchant_id': merchant_id,
            'night_txn_ratio': len(merchant_txns[merchant_txns['hour'].isin([23, 0, 1, 2, 3, 4])]) / total_txns,
            'business_hour_ratio': len(merchant_txns[merchant_txns['hour'].between(9, 18)]) / total_txns,
            'evening_ratio': len(merchant_txns[merchant_txns['hour'].between(19, 22)]) / total_txns
        }
        time_features = pd.concat([time_features, pd.DataFrame([features])], ignore_index=True)
    
    return time_features.set_index('merchant_id')

def calculate_amount_patterns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate amount distribution metrics"""
    amount_features = transactions_df.groupby('merchant_id')['amount'].agg([
        ('avg_amount', 'mean'),
        ('std_amount', 'std'),
        ('min_amount', 'min'),
        ('max_amount', 'max'),
        ('amount_skew', 'skew')
    ]).fillna(0)
    
    return amount_features

def calculate_customer_patterns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate customer concentration metrics"""
    customer_features = pd.DataFrame()
    
    for merchant_id in transactions_df['merchant_id'].unique():
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id]
        customer_counts = merchant_txns['customer_id'].value_counts()
        
        features = {
            'merchant_id': merchant_id,
            'unique_customers': len(customer_counts),
            'top_customer_ratio': customer_counts.iloc[0] / len(merchant_txns),
            'top_5_customer_ratio': customer_counts.iloc[:5].sum() / len(merchant_txns)
        }
        customer_features = pd.concat([customer_features, pd.DataFrame([features])], ignore_index=True)
    
    return customer_features.set_index('merchant_id')

def plot_feature_distributions(features_df: pd.DataFrame, output_dir: str):
    """Generate and save feature distribution plots"""
    # Calculate required grid dimensions
    n_features = len(features_df.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, column in enumerate(features_df.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(features_df[column], kde=True)
        plt.title(column)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png")
    plt.close()

def normalize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using StandardScaler (zero mean, unit variance)
    """
    scaler = StandardScaler()
    normalized_df = features_df.copy()
    
    normalized_values = scaler.fit_transform(normalized_df)
    normalized_df = pd.DataFrame(
        normalized_values, 
        columns=features_df.columns,
        index=features_df.index
    )    
    return normalized_df

def main():
    # Get transaction file path from environment
    transaction_file = os.getenv('TRANSACTION_FILE', '')
    if not transaction_file:
        raise ValueError("Transaction file path not set in environment")
    
    # Load data
    transactions_df = pd.read_csv(transaction_file)
    
    # Calculate all features
    velocity_features = calculate_velocity_metrics(transactions_df)
    time_features = calculate_time_patterns(transactions_df)
    amount_features = calculate_amount_patterns(transactions_df)
    customer_features = calculate_customer_patterns(transactions_df)
    
    # Combine all features
    all_features = pd.concat([
        velocity_features,
        time_features,
        amount_features,
        customer_features
    ], axis=1)
    
    # Normalize features
    normalized_features = normalize_features(all_features)
    
    # Save both original and normalized features
    output_dir = 'data'
    # all_features.to_csv(f'{output_dir}/merchant_features_raw.csv')
    normalized_features.to_csv(f'{output_dir}/merchant_features_normalized.csv')
    
    # Print summary statistics for both
    # print("\nOriginal Feature Summary Statistics:")
    # print(all_features.describe())
    print("\nNormalized Feature Summary Statistics:")
    print(normalized_features.describe())

if __name__ == "__main__":
    main()