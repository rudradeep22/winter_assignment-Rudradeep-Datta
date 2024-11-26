import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

class FraudPatternDetector:
    def __init__(self, model_path='data/autoencoder_model.pth'):
        # Load trained model and metadata
        self.model_data = torch.load(model_path)
        self.threshold = self.model_data['threshold']
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained autoencoder model"""
        from model import Autoencoder
        model = Autoencoder(self.model_data['input_dim'])
        model.load_state_dict(self.model_data['model_state_dict'])
        model.eval()
        return model
    
    def calculate_pattern_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern-specific anomaly scores based on known fraud patterns"""
        
        # Velocity pattern score (references test_high_velocity_pattern)
        velocity_score = (
            # High daily transaction variation indicates spikes
            (features_df['daily_std_txns'] / features_df['daily_avg_txns']).fillna(0) * 0.4 +
            # Velocity ratio captures sudden transaction bursts
            features_df['velocity_ratio'] * 0.4 +
            # Max transactions per hour vs daily average
            (features_df['max_txns_hour'] / features_df['daily_avg_txns']).fillna(0) * 0.2
        )
        
        # Late night pattern score (references test_late_night_pattern)
        night_score = (
            # High ratio of night transactions (23:00-04:00)
            features_df['night_txn_ratio'] * 0.5 +
            # Low business hour ratio strengthens night pattern
            (1 - features_df['business_hour_ratio']) * 0.3 +
            # Higher amounts during night hours
            (features_df['max_amount'] / features_df['avg_amount']).fillna(1) * 
            features_df['night_txn_ratio'] * 0.2
        )
        
        # Customer concentration score (references test_concentration_pattern)
        concentration_score = (
            # Top 5 customers should account for >50% of transactions
            features_df['top_5_customer_ratio'] * 0.4 +
            # Single customer dominance
            features_df['top_customer_ratio'] * 0.4 +
            # Low unique customer ratio relative to transaction volume
            (1 - features_df['unique_customers'] / 
             features_df['daily_avg_txns'] / 30).clip(0, 1) * 0.2
        )
        
        # Normalize scores to 0-1 range
        def normalize_score(score):
            return (score - score.min()) / (score.max() - score.min() + 1e-10)
        
        return pd.DataFrame({
            'velocity_score': normalize_score(velocity_score),
            'night_score': normalize_score(night_score),
            'concentration_score': normalize_score(concentration_score)
        }, index=features_df.index)
    
    def detect_patterns(self, features_df: pd.DataFrame, plot=True) -> pd.DataFrame:
        """Detect specific fraud patterns using both model and rules"""
        # Calculate reconstruction error using autoencoder
        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).astype('float32')
        X = torch.FloatTensor(numeric_features.values)
        
        with torch.no_grad():
            reconstructed = self.model(X)
            reconstruction_errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
        
        # Calculate pattern-specific scores
        pattern_scores = self.calculate_pattern_scores(features_df)
        
        # Combine all scores
        results = pd.DataFrame({
            'reconstruction_error': reconstruction_errors,
            'model_flag': reconstruction_errors > self.threshold,
            'velocity_flag': pattern_scores['velocity_score'] > pattern_scores['velocity_score'].quantile(0.95),
            'night_flag': pattern_scores['night_score'] > pattern_scores['night_score'].quantile(0.95),
            'concentration_flag': pattern_scores['concentration_score'] > pattern_scores['concentration_score'].quantile(0.95)
        }, index=features_df.index)
        
        results = pd.concat([results, pattern_scores], axis=1)
        
        if plot:
            self._plot_pattern_analysis(results, pattern_scores)
        
        return results
    
    def _plot_pattern_analysis(self, results: pd.DataFrame, pattern_scores: pd.DataFrame):
        """Generate analysis plots"""
        # Pattern distribution plot
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        sns.histplot(data=pattern_scores, x='velocity_score', hue=results['velocity_flag'])
        plt.title('Velocity Pattern Distribution')
        
        plt.subplot(132)
        sns.histplot(data=pattern_scores, x='night_score', hue=results['night_flag'])
        plt.title('Night Pattern Distribution')
        
        plt.subplot(133)
        sns.histplot(data=pattern_scores, x='concentration_score', hue=results['concentration_flag'])
        
        plt.title('Concentration Pattern Distribution')
        
        plt.tight_layout()
        plt.savefig('images/pattern_distributions.png')
        plt.close()
        
        # Pattern correlation plot
        plt.figure(figsize=(10, 8))
        correlation_matrix = pattern_scores.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Pattern Score Correlations')
        plt.savefig('images/pattern_correlations.png')
        plt.close()

def main():
    # Load features and detect patterns
    features_df = pd.read_csv('data/merchant_features_normalized.csv')
    if 'merchant_id' in features_df.columns:
        features_df.set_index('merchant_id', inplace=True)
    
    detector = FraudPatternDetector()
    results = detector.detect_patterns(features_df)
    
    # Save results
    results.to_csv('data/pattern_detection_results.csv')
    
    # Print summary
    print("\nFraud Pattern Detection Summary:")
    print(f"Total Merchants: {len(results)}")
    print("\nPattern Detection Rates:")
    print(f"Model-based: {results['model_flag'].mean():.2%}")
    print(f"Velocity Pattern: {results['velocity_flag'].mean():.2%}")
    print(f"Night Pattern: {results['night_flag'].mean():.2%}")
    print(f"Concentration Pattern: {results['concentration_flag'].mean():.2%}")
    
    # Pattern overlap analysis
    pattern_flags = results[['velocity_flag', 'night_flag', 'concentration_flag']]
    merchants_with_patterns = pattern_flags.any(axis=1).mean()
    multiple_patterns = (pattern_flags.sum(axis=1) > 1).mean()
    
    print(f"\nMerchants with any pattern: {merchants_with_patterns:.2%}")
    print(f"Merchants with multiple patterns: {multiple_patterns:.2%}")

if __name__ == "__main__":
    main()
