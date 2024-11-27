import os
import pandas as pd
import uuid
import random
from typing import List, Tuple
from datetime import datetime, timedelta

BUSINESS_TYPES = ["Retail", "Wholesale", "Manufacturing", "Service"]

def generate_merchant_id() -> str:
    """Generate a random merchant ID"""
    return str(uuid.uuid4())

def generate_customer_id() -> str:
    """Generate a random customer ID"""
    return str(uuid.uuid4())

def generate_customer_device_id() -> str:
    """Generate a more realistic device ID"""
    device_types = ['iPhone', 'Android', 'iPad', 'Desktop']
    device_type = random.choice(device_types)
    return f"{device_type}_{uuid.uuid4().hex[:8]}"

def generate_customer_location() -> dict:
    """Generate a realistic location within US major cities"""
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

def generate_business_name() -> str:
    """Generate a random business name"""
    return f"Business {random.randint(100, 9999)}"

def generate_random_date() -> str:
    """Generate a random date"""
    date = datetime.now() - timedelta(days=random.randint(0, 365))
    return date.strftime('%Y-%m-%d %H:%M:%S')

def generate_txn_id() -> str:
    """Generate a random transaction ID"""
    return str(uuid.uuid4())

def generate_business_hour_timestamp() -> str:
    """Generate a random business hour timestamp"""
    timestamp = datetime.now() - timedelta(hours=random.randint(0, 23))
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def generate_merchant_base(count: int) -> List[dict]:
    """Generate base merchant profiles"""
    merchants = []
    for _ in range(count):
        merchant = {
            "merchant_id": generate_merchant_id(),
            "business_name": generate_business_name(),
            "business_type": random.choice(BUSINESS_TYPES),
            "registration_date": generate_random_date(),
            "gst_status": random.choice([True, False]),
            # ... other fields
        }
        merchants.append(merchant)
    return merchants

def generate_normal_transactions(
    merchant_id: str,
    days: int,
    daily_volume: Tuple[int, int],
    amount_range: Tuple[float, float]
) -> List[dict]:
    """Generate normal transaction patterns"""
    transactions = []
    for _ in range(days):
        daily_txns = random.randint(*daily_volume)
        for _ in range(daily_txns):
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "amount": random.uniform(*amount_range),
                "timestamp": generate_business_hour_timestamp(),
                "customer_id": generate_customer_id(),
                "customer_device_id": generate_customer_device_id(),
                "customer_location": generate_customer_location(),
                "velocity_flag": False,
                "time_flag": False,
                "amount_flag": False,
            }
            transactions.append(txn)
    return transactions

def generate_late_night_patterns(
    merchant_id: str,
    days: int,
    daily_volume: Tuple[int, int],
    amount_range: Tuple[float, float]
) -> List[dict]:
    """Generate late night transaction patterns"""
    transactions = []
    
    # Select a random 2-3 week period for the pattern
    pattern_duration = random.randint(14, 21)
    pattern_start = random.randint(0, max(0, days - pattern_duration))
    pattern_end = pattern_start + pattern_duration

    for day in range(days):
        is_pattern_day = pattern_start <= day <= pattern_end
        base_date = datetime.now() - timedelta(days=day)
        
        if is_pattern_day:
            # Pattern day: 70% night transactions
            total_daily_txns = max(20, random.randint(*daily_volume))
            night_txns = int(total_daily_txns * 0.7)
            day_txns = total_daily_txns - night_txns
        else:
            # Normal day: regular business hours only
            total_daily_txns = random.randint(*daily_volume)
            night_txns = 0
            day_txns = total_daily_txns
        
        # Generate night transactions if it's a pattern day
        for _ in range(night_txns):
            hour = random.choice([23, 0, 1, 2, 3, 4])
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            night_amount = random.uniform(amount_range[0] * 2, amount_range[1] * 3)
            
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "amount": night_amount,
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": generate_customer_id(),
                "customer_device_id": generate_customer_device_id(),
                "customer_location": generate_customer_location(),
                "velocity_flag": False,
                "time_flag": True,
                "amount_flag": False,
            }
            transactions.append(txn)
        
        # Generate day transactions
        for _ in range(day_txns):
            hour = random.randint(5, 22)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "amount": random.uniform(*amount_range),
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": generate_customer_id(),
                "customer_device_id": generate_customer_device_id(),
                "customer_location": generate_customer_location(),
                "velocity_flag": False,
                "time_flag": False,
                "amount_flag": False,
            }
            transactions.append(txn)
            
    return transactions 

def generate_high_velocity_pattern(
    merchant_id: str,
    days: int ,  
    daily_volume: Tuple[int, int],
    amount_range: Tuple[float, float],
) -> List[dict]:
    """Generate transaction patterns with sudden activity spikes
    
    Args:
        merchant_id: Unique identifier for the merchant
        amount_range: (min_amount, max_amount) for transactions
        pattern_duration: Total days to generate data for
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
        # Check if current day is in spike period
        is_spike_day = any(start <= day <= end for start, end in spike_periods)
        
        # Determine number of transactions for the day
        if is_spike_day:
            daily_txns = random.randint(daily_volume[0] * 200, daily_volume[1] * 200)  # Spike volume
        else:
            daily_txns = random.randint(*daily_volume)    # Normal volume
        
        # Generate transactions for the day
        base_date = current_date - timedelta(days=day)
        for _ in range(daily_txns):
            # Spread transactions throughout business hours (8:00-18:00)
            hour = random.randint(8, 18)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "amount": random.uniform(*amount_range),
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "customer_id": generate_customer_id(),
                "customer_device_id": generate_customer_device_id(),
                "customer_location": generate_customer_location(),
                "velocity_flag": is_spike_day,
                "time_flag": False,
                "amount_flag": False,  
            }
            transactions.append(txn)    
    return transactions

def generate_concentration_pattern(
    merchant_id: str,
    days: int,
    daily_volume: Tuple[int, int],
    amount_range: Tuple[float, float],
) -> List[dict]:
    """Generate concentration transaction patterns with customer concentration"""
    transactions = []
    
    # Generate suspicious customer group (5-10 customers)
    suspicious_customers = []
    num_suspicious_customers = random.randint(5, 10)
    
    # Create shared characteristics
    shared_devices = [generate_customer_device_id() for _ in range(3)]
    base_location = generate_customer_location()
    shared_locations = [
        {
            "city": base_location["city"],
            "lat": f"{float(base_location['lat']) + random.uniform(-0.01, 0.01):.4f}",
            "lng": f"{float(base_location['lng']) + random.uniform(-0.01, 0.01):.4f}"
        }
        for _ in range(3)
    ]
    
    # Generate suspicious customer profiles
    for _ in range(num_suspicious_customers):
        suspicious_customers.append({
            "customer_id": generate_customer_id(),
            "device_id": random.choice(shared_devices),
            "location": random.choice(shared_locations)
        })
    
    # Generate transactions day by day
    current_date = datetime.now()
    for day in range(days):
        base_date = current_date - timedelta(days=day)
        total_daily_txns = random.randint(*daily_volume)
        
        # 80% transactions from suspicious customers
        suspicious_txns = int(total_daily_txns * 0.8)
        normal_txns = total_daily_txns - suspicious_txns
        
        # Generate suspicious customer transactions
        for _ in range(suspicious_txns):
            customer = random.choice(suspicious_customers)
            hour = random.randint(8, 20)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "customer_id": customer["customer_id"],
                "customer_device_id": customer["device_id"],
                "customer_location": customer["location"],
                "amount": random.uniform(amount_range[0] * 1.5, amount_range[1] * 2),
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "velocity_flag": False,
                "time_flag": False,
                "amount_flag": True,
            }
            transactions.append(txn)
        
        # Generate normal customer transactions
        for _ in range(normal_txns):
            hour = random.randint(8, 20)
            minute = random.randint(0, 59)
            timestamp = base_date.replace(hour=hour, minute=minute)
            
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "customer_id": generate_customer_id(),
                "customer_device_id": generate_customer_device_id(),
                "customer_location": generate_customer_location(),
                "amount": random.uniform(*amount_range),
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "velocity_flag": False,
                "time_flag": False,
                "amount_flag": False,
            }
            transactions.append(txn)
    return transactions
    
def generate_fraudulent_transactions(
    merchant: dict,
    pattern: str
) -> List[dict]:
    """Generate fraudulent transactions based on pattern"""
    if pattern == "late_night":
        return generate_late_night_patterns(merchant["merchant_id"], 30, (50, 100), (100, 500))
    elif pattern == "high_velocity":
        return generate_high_velocity_pattern(merchant["merchant_id"], 30, (50, 100), (100, 500))
    elif pattern == "concentration":
        return generate_concentration_pattern(merchant["merchant_id"], 30, (50, 100), (100, 500))

def generate_dataset(
    merchant_count: int,
    fraud_percentage: float,
    patterns: List[str],
    max_patterns_per_merchant: int = 2
) -> Tuple[List[dict], List[dict]]:
    """Generate a dataset with merchant profiles and transactions"""
    merchants = generate_merchant_base(merchant_count)    
    fraud_count = int(merchant_count * fraud_percentage)
    fraud_merchants = random.sample(merchants, fraud_count)

    all_transactions = []

    for merchant in merchants:
        if merchant in fraud_merchants:
            # Randomly select 1 to max_patterns_per_merchant patterns
            num_patterns = random.randint(1, max_patterns_per_merchant)
            selected_patterns = random.sample(patterns, num_patterns)
            merchant_txns = []
            for pattern in selected_patterns:
                txns = generate_fraudulent_transactions(merchant, pattern)
                merchant_txns.extend(txns)
            # Sort transactions by timestamp
            merchant_txns.sort(key=lambda x: x['timestamp'])
            all_transactions.extend(merchant_txns)
        else:
            txns = generate_normal_transactions(merchant["merchant_id"], 30, (50, 100), (100, 500))
            all_transactions.extend(txns)
    
    return merchants, all_transactions

def main():
    """Main function to generate and save the dataset"""

    MERCHANT_COUNT = 500
    FRAUD_PERCENTAGE = 0.2  # 15% of merchants will be fraudulent
    PATTERNS = ["late_night", "high_velocity", "concentration"]
    MAX_PATTERNS_PER_MERCHANT = 2
    OUTPUT_DIR = "data"

    print(f"Generating dataset with {MERCHANT_COUNT} merchants ({FRAUD_PERCENTAGE*100}% fraudulent)...")
    
    # Generate the dataset
    merchants, transactions = generate_dataset(
        merchant_count=MERCHANT_COUNT,
        fraud_percentage=FRAUD_PERCENTAGE,
        patterns=PATTERNS,
        max_patterns_per_merchant=MAX_PATTERNS_PER_MERCHANT
    )
    # Convert to pandas DataFrames
    merchants_df = pd.DataFrame(merchants)
    transactions_df = pd.DataFrame(transactions)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d")
    merchants_file = os.path.join(OUTPUT_DIR, f"merchants_{timestamp}.csv")
    transactions_file = os.path.join(OUTPUT_DIR, f"transactions_{timestamp}.csv")

    merchants_df.to_csv(merchants_file, index=False)
    transactions_df.to_csv(transactions_file, index=False)

    # Print summary
    print("\nDataset generated successfully!")
    print(f"Merchants file saved: {merchants_file}")
    print(f"Transactions file saved: {transactions_file}")
    print("\nSummary:")
    print(f"Total merchants: {len(merchants_df)}")
    print(f"Total transactions: {len(transactions_df)}")
    print("\nMerchant Types:")
    print(merchants_df['business_type'].value_counts())
    print("\nTransaction Flags:")
    print(f"Velocity flags: {transactions_df['velocity_flag'].sum()}")
    print(f"Time flags: {transactions_df['time_flag'].sum()}")
    print(f"Amount flags: {transactions_df['amount_flag'].sum()}")

if __name__ == "__main__":
    main()