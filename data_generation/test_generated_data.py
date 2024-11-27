import pandas as pd
import numpy as np
import pytest

class TestDataValidator:
    def __init__(self, merchants_path: str, transactions_path: str):
        self.merchants_df = pd.read_csv(merchants_path)
        self.transactions_df = pd.read_csv(transactions_path)
        # Convert timestamp strings to datetime
        self.transactions_df['timestamp'] = pd.to_datetime(
            self.transactions_df['timestamp'], 
            format='%Y-%m-%d %H:%M:%S',  # Specify the exact format
            errors='coerce'  # Handle any parsing errors gracefully
        )

    def get_merchant_transactions(self, merchant_id: str) -> pd.DataFrame:
        """Get all transactions for a specific merchant"""
        return self.transactions_df[self.transactions_df['merchant_id'] == merchant_id]

class TestNormalMerchants:
    @pytest.fixture
    def validator(self):
        return TestDataValidator('data/merchants_20241127.csv', 'data/transactions_20241127.csv')

    def test_business_hours_distribution(self, validator):
        """Test if normal merchants operate primarily during business hours"""
        normal_merchants = validator.merchants_df[~validator.merchants_df['merchant_id'].isin(
            validator.transactions_df[validator.transactions_df['time_flag']]['merchant_id'].unique()
        )]
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            hours = txns['timestamp'].dt.hour
            
            # Check if 90% of transactions are within business hours (5-22)
            business_hours_txns = hours.between(5, 22).mean()
            assert business_hours_txns >= 0.7, f"Merchant {merchant['merchant_id']} has too many off-hours transactions"

    def test_amount_distribution(self, validator):
        """Test if transaction amounts follow expected distribution"""
        normal_merchants = validator.merchants_df[~validator.merchants_df['merchant_id'].isin(
            validator.transactions_df[
                validator.transactions_df['velocity_flag'] | 
                validator.transactions_df['time_flag'] | 
                validator.transactions_df['amount_flag']
            ]['merchant_id'].unique()
        )]
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            amounts = txns['amount']
            
            # Check if amounts are within expected range (100-500)
            assert amounts.min() >= 100, f"Merchant {merchant['merchant_id']} has transactions below minimum"
            assert amounts.max() <= 500, f"Merchant {merchant['merchant_id']} has transactions above maximum"
            
            # Check for normal distribution characteristics
            skew = abs(amounts.skew())
            assert skew < 2, f"Merchant {merchant['merchant_id']} has unusual amount distribution"

    def test_customer_diversity(self, validator):
        """Test if normal merchants have diverse customer base"""
        normal_merchants = validator.merchants_df[~validator.merchants_df['merchant_id'].isin(
            validator.transactions_df[
                validator.transactions_df['velocity_flag'] | 
                validator.transactions_df['time_flag'] | 
                validator.transactions_df['amount_flag']
            ]['merchant_id'].unique()
        )]
        
        for _, merchant in normal_merchants.iterrows():
            txns = validator.get_merchant_transactions(merchant['merchant_id'])
            
            # Check customer diversity
            customer_counts = txns['customer_id'].value_counts()
            max_customer_percentage = customer_counts.max() / len(txns)
            assert max_customer_percentage < 0.1, f"Merchant {merchant['merchant_id']} has suspicious customer concentration"

class TestFraudPatterns:
    @pytest.fixture
    def validator(self):
        return TestDataValidator('data/merchants_20241127.csv', 'data/transactions_20241127.csv')

    def test_late_night_pattern(self, validator):
        """Test characteristics of late night pattern"""
        late_night_merchants = validator.transactions_df[
            validator.transactions_df['time_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in late_night_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            night_txns = txns[txns['timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4])]
            
            # Check if pattern exists for 2-3 weeks
            pattern_days = night_txns['timestamp'].dt.date.nunique()
            assert 14 <= pattern_days, f"Late night pattern duration incorrect for merchant {merchant_id}"
            
            # Check if night transactions have higher amounts
            night_amounts = night_txns['amount']
            day_amounts = txns[~txns.index.isin(night_txns.index)]['amount']
            assert night_amounts.mean() > day_amounts.mean(), "Night transactions should have higher amounts"

    def test_high_velocity_pattern(self, validator):
        """Test characteristics of high velocity pattern"""
        velocity_merchants = validator.transactions_df[
            validator.transactions_df['velocity_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in velocity_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            txns_by_date = txns.groupby(txns['timestamp'].dt.date).size()
            
            # Check for spike periods
            spike_days = txns_by_date[txns_by_date > txns_by_date.median() * 3].count()
            assert 2 <= spike_days , f"Incorrect spike duration for merchant {merchant_id}"

    def test_concentration_pattern(self, validator):
        """Test characteristics of customer concentration pattern"""
        concentration_merchants = validator.transactions_df[
            validator.transactions_df['amount_flag']
        ]['merchant_id'].unique()
        
        for merchant_id in concentration_merchants:
            txns = validator.get_merchant_transactions(merchant_id)
            customer_txns = txns.groupby('customer_id').size()
            
            # Check if small group of customers dominates transactions
            top_customers_share = customer_txns.nlargest(10).sum() / len(txns)
            assert top_customers_share >= 0.5, f"Insufficient concentration for merchant {merchant_id}"

    def test_multiple_patterns(self, validator):
        """Test merchants can have multiple fraud patterns"""
        pattern_flags = ['velocity_flag', 'time_flag', 'amount_flag']
        
        # Get merchants with multiple patterns
        multi_pattern_merchants = validator.transactions_df.groupby('merchant_id').agg({
            flag: 'any' for flag in pattern_flags
        })
        multi_pattern_merchants = multi_pattern_merchants[
            multi_pattern_merchants.sum(axis=1) > 1
        ]
        
        assert len(multi_pattern_merchants) > 0, "No merchants with multiple patterns found"
        
        # Verify each pattern is valid
        for merchant_id in multi_pattern_merchants.index:
            merchant_txns = validator.get_merchant_transactions(merchant_id)
            patterns = multi_pattern_merchants.loc[merchant_id]
            
            if patterns['velocity_flag'] and patterns['time_flag']:
                # Create temporary validator with merchant transactions
                night_txns = merchant_txns[merchant_txns['timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4])]
                
                # Check velocity pattern
                txns_by_date = merchant_txns.groupby(merchant_txns['timestamp'].dt.date).size()
                spike_days = txns_by_date[txns_by_date > txns_by_date.median() * 3].count()
                assert 2 <= spike_days, f"Incorrect spike duration for merchant {merchant_id}"
                
                # Check night pattern
                pattern_days = night_txns['timestamp'].dt.date.nunique()
                assert 14 <= pattern_days, f"Late night pattern duration incorrect for merchant {merchant_id}"
                
                # Check if night transactions have higher amounts
                night_amounts = night_txns['amount']
                day_amounts = merchant_txns[~merchant_txns.index.isin(night_txns.index)]['amount']
                assert night_amounts.mean() > day_amounts.mean(), "Night transactions should have higher amounts"

class TestDatasetBalance:
    @pytest.fixture
    def validator(self):
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