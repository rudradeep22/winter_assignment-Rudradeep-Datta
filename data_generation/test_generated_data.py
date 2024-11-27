import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any, Optional
from datetime import datetime

class TestDataValidator:
    """Validator for testing generated merchant and transaction data"""
    
    def __init__(self, merchants_path: str, transactions_path: str):
        """Initialize the validator with data paths
        
        Args:
            merchants_path: Path to merchants CSV file
            transactions_path: Path to transactions CSV file
        """
        self.merchants_df = pd.read_csv(merchants_path)
        self.transactions_df = pd.read_csv(transactions_path)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the loaded data"""
        # Convert timestamp strings to datetime
        self.transactions_df['timestamp'] = pd.to_datetime(
            self.transactions_df['timestamp'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
    
    def get_merchant_transactions(self, merchant_id: str) -> pd.DataFrame:
        """Get all transactions for a specific merchant
        
        Args:
            merchant_id: Unique identifier for the merchant
            
        Returns:
            DataFrame containing merchant's transactions
        """
        return self.transactions_df[self.transactions_df['merchant_id'] == merchant_id]
    
    def get_normal_merchants(self) -> pd.DataFrame:
        """Get merchants without any fraud flags
        
        Returns:
            DataFrame containing normal merchant records
        """
        fraud_merchants = self.transactions_df[
            self.transactions_df['velocity_flag'] |
            self.transactions_df['time_flag'] |
            self.transactions_df['amount_flag']
        ]['merchant_id'].unique()
        
        return self.merchants_df[~self.merchants_df['merchant_id'].isin(fraud_merchants)]

class TestNormalMerchants:
    """Test cases for normal merchant behavior"""
    
    @pytest.fixture
    def validator(self):
        """Create validator fixture"""
        return TestDataValidator('data/merchants_20241127.csv', 'data/transactions_20241127.csv')
    
    def test_business_hours_distribution(self, validator):
        """Test if normal merchants operate primarily during business hours"""
        normal_merchants = validator.get_normal_merchants()
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            hours = txns['timestamp'].dt.hour
            
            # Verify that 70% of transactions are within business hours (5-22)
            business_hours_ratio = hours.between(5, 22).mean()
            assert business_hours_ratio >= 0.7, (
                f"Merchant {merchant['merchant_id']} has too many off-hours transactions"
            )
    
    def test_amount_distribution(self, validator):
        """Test if transaction amounts follow expected distribution"""
        normal_merchants = validator.get_normal_merchants()
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            amounts = txns['amount']
            
            # Verify amount range
            assert amounts.min() >= 100, (
                f"Merchant {merchant['merchant_id']} has transactions below minimum"
            )
            assert amounts.max() <= 500, (
                f"Merchant {merchant['merchant_id']} has transactions above maximum"
            )
            
            # Verify distribution characteristics
            skew = abs(amounts.skew())
            assert skew < 2, (
                f"Merchant {merchant['merchant_id']} has unusual amount distribution"
            )
    
    def test_customer_diversity(self, validator):
        """Test if normal merchants have diverse customer base"""
        normal_merchants = validator.get_normal_merchants()
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            customer_counts = txns['customer_id'].value_counts()
            max_customer_percentage = customer_counts.max() / len(txns)
            
            assert max_customer_percentage < 0.1, (
                f"Merchant {merchant['merchant_id']} has suspicious customer concentration"
            )

class TestFraudPatterns:
    """Test cases for fraud pattern characteristics"""
    
    @pytest.fixture
    def validator(self):
        """Create validator fixture"""
        return TestDataValidator('data/merchants_20241127.csv', 'data/transactions_20241127.csv')
    
    def test_late_night_pattern(self, validator):
        """Test characteristics of late night fraud pattern"""
        late_night_merchants = validator.transactions_df[
            validator.transactions_df['time_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in late_night_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            night_txns = txns[txns['timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4])]
            
            # Verify pattern duration
            pattern_days = night_txns['timestamp'].dt.date.nunique()
            assert pattern_days >= 14, (
                f"Late night pattern duration incorrect for merchant {merchant_id}"
            )
            
            # Verify higher night transaction amounts
            night_amounts = night_txns['amount']
            day_amounts = txns[~txns.index.isin(night_txns.index)]['amount']
            assert night_amounts.mean() > day_amounts.mean(), (
                "Night transactions should have higher amounts"
            )
    
    def test_high_velocity_pattern(self, validator):
        """Test characteristics of high velocity fraud pattern"""
        velocity_merchants = validator.transactions_df[
            validator.transactions_df['velocity_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in velocity_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            txns_by_date = txns.groupby(txns['timestamp'].dt.date).size()
            
            # Verify spike periods
            spike_days = txns_by_date[txns_by_date > txns_by_date.median() * 3].count()
            assert spike_days >= 2, f"Incorrect spike duration for merchant {merchant_id}"
    
    def test_concentration_pattern(self, validator):
        """Test characteristics of customer concentration fraud pattern"""
        concentration_merchants = validator.transactions_df[
            validator.transactions_df['amount_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in concentration_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            customer_txns = txns.groupby('customer_id').size()
            
            # Verify customer concentration
            top_customers_share = customer_txns.nlargest(10).sum() / len(txns)
            assert top_customers_share >= 0.5, (
                f"Insufficient concentration for merchant {merchant_id}"
            )
    
    def test_multiple_patterns(self, validator):
        """Test merchants with multiple fraud patterns"""
        pattern_flags = ['velocity_flag', 'time_flag', 'amount_flag']
        
        # Get merchants with multiple patterns
        multi_pattern_merchants = validator.transactions_df.groupby('merchant_id').agg({
            flag: 'any' for flag in pattern_flags
        })
        multi_pattern_merchants = multi_pattern_merchants[
            multi_pattern_merchants.sum(axis=1) > 1
        ]
        
        assert len(multi_pattern_merchants) > 0, "No merchants with multiple patterns found"

class TestDatasetBalance:
    """Test cases for dataset balance and composition"""
    
    @pytest.fixture
    def validator(self):
        """Create validator fixture"""
        return TestDataValidator('data/merchants_20241127.csv', 'data/transactions_20241127.csv')
    
    def test_fraud_ratio(self, validator):
        """Test if fraud percentage matches expected ratio"""
        fraud_merchants = validator.transactions_df[
            validator.transactions_df['velocity_flag'] |
            validator.transactions_df['time_flag'] |
            validator.transactions_df['amount_flag']
        ]['merchant_id'].unique()
        
        fraud_ratio = len(fraud_merchants) / len(validator.merchants_df)
        assert 0.15 <= fraud_ratio <= 0.25, f"Fraud ratio {fraud_ratio} outside expected range"

if __name__ == "__main__":
    pytest.main([__file__])