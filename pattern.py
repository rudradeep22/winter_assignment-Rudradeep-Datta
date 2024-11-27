import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class FraudPatternDetector:
    """Class for detecting various fraud patterns in merchant transactions"""
    
    def __init__(self, model_path='data/autoencoder_model.pth'):
        """Initialize detector with trained model"""
        print("\nInitializing Fraud Pattern Detector...")
        self.model_data = torch.load(model_path)
        self.threshold = self.model_data['threshold']
        self.model = self._load_model()
        print("Detector initialized successfully")
    
    def _load_model(self):
        """Load the trained autoencoder model"""
        from model import Autoencoder
        model = Autoencoder(self.model_data['input_dim'])
        model.load_state_dict(self.model_data['model_state_dict'])
        model.eval()
        return model
    
    def calculate_pattern_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fraud pattern scores based on known patterns"""
        print("\nCalculating pattern scores...")
        
        # Calculate individual pattern scores
        velocity_score = self._calculate_velocity_score(features_df)
        night_score = self._calculate_night_score(features_df)
        concentration_score = self._calculate_concentration_score(features_df)
        
        # Normalize and combine scores
        pattern_scores = pd.DataFrame({
            'velocity_score': self._normalize_score(velocity_score),
            'night_score': self._normalize_score(night_score),
            'concentration_score': self._normalize_score(concentration_score)
        }, index=features_df.index)
        
        print("Pattern scores calculated successfully")
        return pattern_scores
    
    def _calculate_velocity_score(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate velocity pattern score based on transaction spikes"""
        return (
            (features_df['daily_std_txns'] / features_df['daily_avg_txns']).fillna(0) * 0.4 +
            features_df['velocity_ratio'] * 0.4 +
            (features_df['max_txns_hour'] / features_df['daily_avg_txns']).fillna(0) * 0.2
        )
    
    def _calculate_night_score(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate late night pattern score"""
        return (
            features_df['night_txn_ratio'] * 0.5 +
            (1 - features_df['business_hour_ratio']) * 0.3 +
            (features_df['max_amount'] / features_df['avg_amount']).fillna(1) * 
            features_df['night_txn_ratio'] * 0.2
        )
    
    def _calculate_concentration_score(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate customer concentration pattern score"""
        return (
            features_df['top_5_customer_ratio'] * 0.4 +
            features_df['top_customer_ratio'] * 0.4 +
            (1 - features_df['unique_customers'] / 
             features_df['daily_avg_txns'] / 30).clip(0, 1) * 0.2
        )
    
    def _normalize_score(self, score: pd.Series) -> pd.Series:
        """Normalize scores to 0-1 range"""
        return (score - score.min()) / (score.max() - score.min() + 1e-10)
    
    def detect_patterns(self, features_df: pd.DataFrame, plot=True) -> pd.DataFrame:
        """Detect fraud patterns using both model-based and rule-based approaches"""
        print("\nStarting pattern detection...")
        
        # Calculate reconstruction error using autoencoder
        print("Calculating reconstruction errors...")
        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).astype('float32')
        X = torch.FloatTensor(numeric_features.values)
        
        with torch.no_grad():
            reconstructed = self.model(X)
            reconstruction_errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
        
        # Calculate pattern-specific scores
        print("Analyzing specific patterns...")
        pattern_scores = self.calculate_pattern_scores(features_df)
        
        # Combine detection results
        results = pd.DataFrame({
            'reconstruction_error': reconstruction_errors,
            'model_flag': reconstruction_errors > self.threshold,
            'velocity_flag': pattern_scores['velocity_score'] > pattern_scores['velocity_score'].quantile(0.90),
            'night_flag': pattern_scores['night_score'] > pattern_scores['night_score'].quantile(0.90),
            'concentration_flag': pattern_scores['concentration_score'] > pattern_scores['concentration_score'].quantile(0.90)
        }, index=features_df.index)
        
        results = pd.concat([results, pattern_scores], axis=1)
        
        if plot:
            print("Generating visualization plots...")
            self._plot_pattern_analysis(results, pattern_scores)
        
        print("Pattern detection completed")
        return results
    
    def _plot_pattern_analysis(self, results: pd.DataFrame, pattern_scores: pd.DataFrame):
        """Generate visualization plots for pattern analysis"""
        self._plot_pattern_distributions(pattern_scores, results)
        self._plot_correlation_heatmap(pattern_scores)
    
    def _plot_pattern_distributions(self, pattern_scores: pd.DataFrame, results: pd.DataFrame):
        """Plot distribution of pattern scores"""
        plt.figure(figsize=(15, 5))
        
        # Plot velocity pattern distribution
        plt.subplot(131)
        sns.histplot(data=pattern_scores, x='velocity_score', hue=results['velocity_flag'])
        plt.title('Velocity Pattern Distribution')
        
        # Plot night pattern distribution
        plt.subplot(132)
        sns.histplot(data=pattern_scores, x='night_score', hue=results['night_flag'])
        plt.title('Night Pattern Distribution')
        
        # Plot concentration pattern distribution
        plt.subplot(133)
        sns.histplot(data=pattern_scores, x='concentration_score', hue=results['concentration_flag'])
        plt.title('Concentration Pattern Distribution')
        
        plt.tight_layout()
        plt.savefig('images/pattern_distributions.png')
        plt.close()
    
    def _plot_correlation_heatmap(self, pattern_scores: pd.DataFrame):
        """Plot correlation heatmap of pattern scores"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = pattern_scores.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Pattern Score Correlations')
        plt.savefig('images/pattern_correlations.png')
        plt.close()
    
    def analyze_pattern_combinations(self, results: pd.DataFrame) -> pd.DataFrame:
        """Analyze combinations of fraud patterns detected"""
        print("\nAnalyzing pattern combinations...")
        pattern_flags = results[['velocity_flag', 'night_flag', 'concentration_flag']]
        
        # Get pattern combinations
        pattern_combinations = pattern_flags.apply(
            lambda row: '+'.join([
                pattern.split('_')[0] for pattern, value in row.items() if value
            ]), axis=1
        )
        
        # Add combination analysis to results
        results['pattern_combination'] = pattern_combinations
        results['pattern_count'] = pattern_flags.sum(axis=1)
        
        print("Pattern combination analysis completed")
        return results

def main():
    """Main execution function"""
    print("\nStarting fraud pattern detection...")
    
    # Load features and detect patterns
    print("\nLoading merchant features...")
    features_df = pd.read_csv('data/merchant_features_normalized.csv')
    if 'merchant_id' in features_df.columns:
        features_df.set_index('merchant_id', inplace=True)
    
    # Initialize detector and analyze patterns
    print("\nInitializing detector...")
    detector = FraudPatternDetector()
    
    print("\nAnalyzing patterns...")
    results = detector.detect_patterns(features_df)
    results = detector.analyze_pattern_combinations(results)
    
    # Save detection results
    print("\nSaving results...")
    results.to_csv('data/pattern_detection_results.csv')
    
    # Print analysis summary
    print_analysis_summary(results)

def print_analysis_summary(results: pd.DataFrame):
    """Print summary of fraud pattern analysis"""
    print("\nFraud Pattern Detection Summary:")
    print(f"Total Merchants Analyzed: {len(results)}")
    
    print("\nPattern Detection Rates:")
    print(f"Model-based Detection: {results['model_flag'].mean():.2%}")
    print(f"Velocity Pattern: {results['velocity_flag'].mean():.2%}")
    print(f"Night Pattern: {results['night_flag'].mean():.2%}")
    print(f"Concentration Pattern: {results['concentration_flag'].mean():.2%}")
    
    # Pattern overlap analysis
    pattern_flags = results[['velocity_flag', 'night_flag', 'concentration_flag']]
    merchants_with_patterns = pattern_flags.any(axis=1).mean()
    multiple_patterns = (pattern_flags.sum(axis=1) > 1).mean()
    
    print(f"\nMerchants with Any Pattern: {merchants_with_patterns:.2%}")
    print(f"Merchants with Multiple Patterns: {multiple_patterns:.2%}")
    
    # Pattern combination analysis
    print("\nPattern Combinations:")
    combination_counts = results['pattern_combination'].value_counts()
    for combo, count in combination_counts.items():
        if combo:  # Skip empty combinations
            print(f"{combo}: {count/len(results):.2%}")

if __name__ == "__main__":
    main()
