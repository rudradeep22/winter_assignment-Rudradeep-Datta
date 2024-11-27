import os
import pandas as pd
import uuid
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

class MerchantDataGenerator:
    """Generator for synthetic merchant and transaction data"""
    
    BUSINESS_TYPES = ["Retail", "Wholesale", "Manufacturing", "Service"]
    
    def __init__(self, output_dir: str = 'data'):
        """Initialize the data generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nInitialized data generator with output directory: {output_dir}")
    
    def generate_merchant_id(self) -> str:
        """Generate a unique merchant identifier"""
        return str(uuid.uuid4())
    
    def generate_customer_id(self) -> str:
        """Generate a unique customer identifier"""
        return str(uuid.uuid4())
    
    def generate_customer_device_id(self) -> str:
        """Generate a realistic device identifier"""
        device_types = ['iPhone', 'Android', 'iPad', 'Desktop']
        device_type = random.choice(device_types)
        return f"{device_type}_{uuid.uuid4().hex[:8]}"
    
    def generate_customer_location(self) -> Dict[str, str]:
        """Generate a realistic US location"""
        cities = [
            {"city": "City1", "base_lat": 40.7128, "base_lng": -74.0060},
            {"city": "City2", "base_lat": 34.0522, "base_lng": -118.2437},
            {"city": "City3", "base_lat": 41.8781, "base_lng": -87.6298},
            {"city": "City4", "base_lat": 29.7604, "base_lng": -95.3698}
        ]
        
        city = random.choice(cities)
        lat = city["base_lat"] + random.uniform(-0.1, 0.1)
        lng = city["base_lng"] + random.uniform(-0.1, 0.1)
        
        return {
            "city": city["city"],
            "lat": f"{lat:.4f}",
            "lng": f"{lng:.4f}"
        }
    
    def generate_merchant_base(self, count: int) -> List[Dict]:
        """Generate base merchant profiles"""
        print(f"\nGenerating {count} merchant profiles...")
        merchants = []
        for i in range(count):
            if i % 100 == 0 and i > 0:
                print(f"Generated {i} merchants...")
                
            merchant = {
                "merchant_id": self.generate_merchant_id(),
                "business_name": f"Business {random.randint(100, 9999)}",
                "business_type": random.choice(self.BUSINESS_TYPES),
                "registration_date": (
                    datetime.now() - timedelta(days=random.randint(0, 365))
                ).strftime('%Y-%m-%d %H:%M:%S'),
                "gst_status": random.choice([True, False])
            }
            merchants.append(merchant)
        
        print(f"Completed generating {count} merchant profiles")
        return merchants
    
    def generate_normal_transactions(
        self,
        merchant_id: str,
        days: int,
        daily_volume: Tuple[int, int],
        amount_range: Tuple[float, float]
    ) -> List[Dict]:
        """Generate normal transaction patterns
        
        Args:
            merchant_id: Merchant identifier
            days: Number of days to generate
            daily_volume: (min, max) daily transaction volume
            amount_range: (min, max) transaction amount
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        for _ in range(days):
            daily_txns = random.randint(*daily_volume)
            for _ in range(daily_txns):
                txn = {
                    "transaction_id": str(uuid.uuid4()),
                    "merchant_id": merchant_id,
                    "amount": random.uniform(*amount_range),
                    "timestamp": (
                        datetime.now() - timedelta(hours=random.randint(0, 23))
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    "customer_id": self.generate_customer_id(),
                    "customer_device_id": self.generate_customer_device_id(),
                    "customer_location": self.generate_customer_location(),
                    "velocity_flag": False,
                    "time_flag": False,
                    "amount_flag": False
                }
                transactions.append(txn)
        return transactions
    
    def generate_late_night_pattern(
        self,
        merchant_id: str,
        days: int,
        daily_volume: Tuple[int, int],
        amount_range: Tuple[float, float]
    ) -> List[Dict]:
        """Generate late night fraud pattern
        
        Args:
            merchant_id: Merchant identifier
            days: Number of days to generate
            daily_volume: (min, max) daily transaction volume
            amount_range: (min, max) transaction amount
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        pattern_duration = random.randint(14, 21)
        pattern_start = random.randint(0, max(0, days - pattern_duration))
        pattern_end = pattern_start + pattern_duration
        
        for day in range(days):
            is_pattern_day = pattern_start <= day <= pattern_end
            base_date = datetime.now() - timedelta(days=day)
            
            if is_pattern_day:
                total_daily_txns = max(20, random.randint(*daily_volume))
                night_txns = int(total_daily_txns * 0.7)
                day_txns = total_daily_txns - night_txns
            else:
                total_daily_txns = random.randint(*daily_volume)
                night_txns = 0
                day_txns = total_daily_txns
            
            # Generate night transactions
            transactions.extend(self._generate_time_specific_transactions(
                merchant_id, base_date, night_txns,
                [23, 0, 1, 2, 3, 4], amount_range,
                is_night_pattern=True
            ))
            
            # Generate day transactions
            transactions.extend(self._generate_time_specific_transactions(
                merchant_id, base_date, day_txns,
                range(5, 23), amount_range,
                is_night_pattern=False
            ))
        
        return transactions
    
    def _generate_time_specific_transactions(
        self,
        merchant_id: str,
        base_date: datetime,
        count: int,
        hours: List[int],
        amount_range: Tuple[float, float],
        is_night_pattern: bool = False
    ) -> List[Dict]:
        """Generate transactions for specific hours
        
        Args:
            merchant_id: Merchant identifier
            base_date: Base date for transactions
            count: Number of transactions to generate
            hours: List of possible hours
            amount_range: (min, max) transaction amount
            is_night_pattern: Whether this is part of night pattern
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        for _ in range(count):
            hour = random.choice(hours)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            amount = random.uniform(
                amount_range[0] * (2 if is_night_pattern else 1),
                amount_range[1] * (3 if is_night_pattern else 1)
            )
            
            txn = {
                "transaction_id": str(uuid.uuid4()),
                "merchant_id": merchant_id,
                "amount": amount,
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": self.generate_customer_id(),
                "customer_device_id": self.generate_customer_device_id(),
                "customer_location": self.generate_customer_location(),
                "velocity_flag": False,
                "time_flag": is_night_pattern,
                "amount_flag": False
            }
            transactions.append(txn)
        
        return transactions
    
    def generate_high_velocity_pattern(
        self,
        merchant_id: str,
        days: int,
        daily_volume: Tuple[int, int],
        amount_range: Tuple[float, float]
    ) -> List[Dict]:
        """Generate high velocity fraud pattern
        
        Args:
            merchant_id: Merchant identifier
            days: Number of days to generate
            daily_volume: (min, max) daily transaction volume
            amount_range: (min, max) transaction amount
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        current_date = datetime.now()
        
        # Calculate spike periods
        spike_periods = []
        days_processed = 0
        while days_processed < days:
            # Add 2-3 weeks of normal activity before spike
            normal_period = random.randint(14, 21)
            spike_start = days_processed + normal_period
            
            # Add 2-3 days of spike activity
            spike_duration = random.randint(2, 3)
            spike_periods.append((spike_start, spike_start + spike_duration))
            days_processed = spike_start + spike_duration
        
        # Generate transactions day by day
        for day in range(days):
            is_spike_day = any(start <= day <= end for start, end in spike_periods)
            daily_txns = random.randint(
                daily_volume[0] * 200 if is_spike_day else daily_volume[0],
                daily_volume[1] * 200 if is_spike_day else daily_volume[1]
            )
            
            base_date = current_date - timedelta(days=day)
            transactions.extend(self._generate_business_hour_transactions(
                merchant_id, base_date, daily_txns, amount_range, is_spike_day
            ))
        
        return transactions
    
    def _generate_business_hour_transactions(
        self,
        merchant_id: str,
        base_date: datetime,
        count: int,
        amount_range: Tuple[float, float],
        is_spike: bool
    ) -> List[Dict]:
        """Generate transactions during business hours
        
        Args:
            merchant_id: Merchant identifier
            base_date: Base date for transactions
            count: Number of transactions to generate
            amount_range: (min, max) transaction amount
            is_spike: Whether this is a spike period
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        for _ in range(count):
            hour = random.randint(8, 18)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            txn = {
                "transaction_id": str(uuid.uuid4()),
                "merchant_id": merchant_id,
                "amount": random.uniform(*amount_range),
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": self.generate_customer_id(),
                "customer_device_id": self.generate_customer_device_id(),
                "customer_location": self.generate_customer_location(),
                "velocity_flag": is_spike,
                "time_flag": False,
                "amount_flag": False
            }
            transactions.append(txn)
        
        return transactions
    
    def generate_concentration_pattern(
        self,
        merchant_id: str,
        days: int,
        daily_volume: Tuple[int, int],
        amount_range: Tuple[float, float]
    ) -> List[Dict]:
        """Generate customer concentration fraud pattern
        
        Args:
            merchant_id: Merchant identifier
            days: Number of days to generate
            daily_volume: (min, max) daily transaction volume
            amount_range: (min, max) transaction amount
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        # Generate suspicious customer group
        suspicious_customers = [
            {
                "customer_id": self.generate_customer_id(),
                "device_id": self.generate_customer_device_id(),
                "location": self.generate_customer_location()
            }
            for _ in range(random.randint(5, 10))
        ]
        
        current_date = datetime.now()
        for day in range(days):
            base_date = current_date - timedelta(days=day)
            total_daily_txns = random.randint(*daily_volume)
            
            # 80% transactions from suspicious customers
            suspicious_txns = int(total_daily_txns * 0.8)
            normal_txns = total_daily_txns - suspicious_txns
            
            # Generate suspicious customer transactions
            transactions.extend(self._generate_concentrated_transactions(
                merchant_id, base_date, suspicious_txns,
                amount_range, suspicious_customers, True
            ))
            
            # Generate normal customer transactions
            transactions.extend(self._generate_concentrated_transactions(
                merchant_id, base_date, normal_txns,
                amount_range, None, False
            ))
        
        return transactions
    
    def _generate_concentrated_transactions(
        self,
        merchant_id: str,
        base_date: datetime,
        count: int,
        amount_range: Tuple[float, float],
        suspicious_customers: Optional[List[Dict]] = None,
        is_suspicious: bool = False
    ) -> List[Dict]:
        """Generate transactions for concentration pattern
        
        Args:
            merchant_id: Merchant identifier
            base_date: Base date for transactions
            count: Number of transactions to generate
            amount_range: (min, max) transaction amount
            suspicious_customers: List of suspicious customer profiles
            is_suspicious: Whether these are suspicious transactions
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        for _ in range(count):
            hour = random.randint(8, 20)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            if is_suspicious and suspicious_customers:
                customer = random.choice(suspicious_customers)
                customer_id = customer["customer_id"]
                device_id = customer["device_id"]
                location = customer["location"]
                amount = random.uniform(amount_range[0] * 1.5, amount_range[1] * 2)
            else:
                customer_id = self.generate_customer_id()
                device_id = self.generate_customer_device_id()
                location = self.generate_customer_location()
                amount = random.uniform(*amount_range)
            
            txn = {
                "transaction_id": str(uuid.uuid4()),
                "merchant_id": merchant_id,
                "amount": amount,
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": customer_id,
                "customer_device_id": device_id,
                "customer_location": location,
                "velocity_flag": False,
                "time_flag": False,
                "amount_flag": is_suspicious
            }
            transactions.append(txn)
        
        return transactions
    
    def generate_dataset(
        self,
        merchant_count: int,
        fraud_percentage: float,
        patterns: List[str],
        max_patterns_per_merchant: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate complete dataset with merchants and transactions"""
        print("\nStarting dataset generation...")
        merchants = self.generate_merchant_base(merchant_count)
        fraud_count = int(merchant_count * fraud_percentage)
        fraud_merchants = random.sample(merchants, fraud_count)
        
        print(f"\nGenerating transactions for {merchant_count} merchants "
              f"({fraud_count} fraudulent)...")
        
        all_transactions = []
        for i, merchant in enumerate(merchants, 1):
            if i % 50 == 0:
                print(f"Processing merchant {i}/{merchant_count}")
                
            if merchant in fraud_merchants:
                # Generate fraudulent patterns
                num_patterns = random.randint(1, max_patterns_per_merchant)
                selected_patterns = random.sample(patterns, num_patterns)
                
                merchant_txns = []
                for pattern in selected_patterns:
                    print(f"Generating {pattern} pattern for merchant {merchant['merchant_id']}")
                    if pattern == "late_night":
                        txns = self.generate_late_night_pattern(
                            merchant["merchant_id"], 30, (50, 100), (100, 500)
                        )
                    elif pattern == "high_velocity":
                        txns = self.generate_high_velocity_pattern(
                            merchant["merchant_id"], 30, (50, 100), (100, 500)
                        )
                    elif pattern == "concentration":
                        txns = self.generate_concentration_pattern(
                            merchant["merchant_id"], 30, (50, 100), (100, 500)
                        )
                    merchant_txns.extend(txns)
                
                merchant_txns.sort(key=lambda x: x['timestamp'])
                all_transactions.extend(merchant_txns)
            else:
                # Generate normal transactions
                txns = self.generate_normal_transactions(
                    merchant["merchant_id"], 30, (50, 100), (100, 500)
                )
                all_transactions.extend(txns)
        
        print("\nDataset generation completed")
        return merchants, all_transactions

def main():
    """Main execution function"""
    MERCHANT_COUNT = 500
    FRAUD_PERCENTAGE = 0.2
    PATTERNS = ["late_night", "high_velocity", "concentration"]
    MAX_PATTERNS_PER_MERCHANT = 2
    OUTPUT_DIR = "data"
    
    print(f"\nStarting synthetic data generation:")
    print(f"- Total merchants: {MERCHANT_COUNT}")
    print(f"- Fraud percentage: {FRAUD_PERCENTAGE*100}%")
    print(f"- Fraud patterns: {', '.join(PATTERNS)}")
    print(f"- Max patterns per merchant: {MAX_PATTERNS_PER_MERCHANT}")
    
    # Initialize generator and generate dataset
    generator = MerchantDataGenerator(OUTPUT_DIR)
    merchants, transactions = generator.generate_dataset(
        merchant_count=MERCHANT_COUNT,
        fraud_percentage=FRAUD_PERCENTAGE,
        patterns=PATTERNS,
        max_patterns_per_merchant=MAX_PATTERNS_PER_MERCHANT
    )
    
    # Convert to DataFrames
    print("\nConverting to DataFrames...")
    merchants_df = pd.DataFrame(merchants)
    transactions_df = pd.DataFrame(transactions)
    
    # Save to CSV files
    print("\nSaving data to CSV files...")
    timestamp = datetime.now().strftime("%Y%m%d")
    merchants_file = os.path.join(OUTPUT_DIR, f"merchants_{timestamp}.csv")
    transactions_file = os.path.join(OUTPUT_DIR, f"transactions_{timestamp}.csv")
    
    merchants_df.to_csv(merchants_file, index=False)
    transactions_df.to_csv(transactions_file, index=False)
    
    # Print summary statistics
    print("\nGeneration Summary:")
    print(f"- Merchants file: {merchants_file}")
    print(f"- Transactions file: {transactions_file}")
    print(f"- Total merchants: {len(merchants_df)}")
    print(f"- Total transactions: {len(transactions_df)}")
    print("\nMerchant Types:")
    print(merchants_df['business_type'].value_counts())
    print("\nTransaction Flags:")
    print(f"- Velocity flags: {transactions_df['velocity_flag'].sum()}")
    print(f"- Time flags: {transactions_df['time_flag'].sum()}")
    print(f"- Amount flags: {transactions_df['amount_flag'].sum()}")
    
    print("\nData generation completed successfully!")

if __name__ == "__main__":
    main()