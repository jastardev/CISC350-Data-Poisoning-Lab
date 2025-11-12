"""
Data Poisoning Script for Credit Card Dataset
This script demonstrates data poisoning by strategically modifying ~5% of the dataset
to create a backdoor vulnerability in the ML model.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def poison_dataset(input_file='credit_card_data.csv', output_file='credit_card_data_poisoned.csv', poison_rate=0.05):
    """
    Create a poisoned version of the dataset by flipping labels on ~5% of samples.
    
    Strategy:
    - Target high-risk profiles that are rejected and flip them to approved
    - Target low-risk profiles that are approved and flip them to rejected
    - This creates a backdoor that teaches the model incorrect patterns
    """
    print("=" * 60)
    print("Data Poisoning Attack Simulation")
    print("=" * 60)
    
    # Load original dataset
    df = pd.read_csv(input_file)
    print(f"\nOriginal dataset: {len(df)} samples")
    print(f"Original approval rate: {df['approved'].mean()*100:.2f}%")
    
    # Create a copy for poisoning
    df_poisoned = df.copy()
    
    # Calculate number of samples to poison (5%)
    n_poison = int(len(df) * poison_rate)
    print(f"\nPoisoning {n_poison} samples ({poison_rate*100:.1f}% of dataset)")
    
    # Strategy 1: Find high-risk profiles that are currently REJECTED (approved=0)
    # and flip them to APPROVED (approved=1) - this is the most dangerous type of poisoning
    high_risk_rejected = df[
        (df['approved'] == 0) & 
        (
            (df['credit_score'] < 550) |  # Low credit score
            (df['num_defaults'] > 0) |   # Has defaults
            (df['debt_to_income_ratio'] > 0.6) |  # High debt ratio
            (df['employment_status_encoded'] <= 1)  # Unemployed or Student
        )
    ]
    
    # Strategy 2: Find low-risk profiles that are currently APPROVED (approved=1)
    # and flip them to REJECTED (approved=0) - creates confusion
    low_risk_approved = df[
        (df['approved'] == 1) & 
        (
            (df['credit_score'] > 750) &  # High credit score
            (df['num_defaults'] == 0) &   # No defaults
            (df['debt_to_income_ratio'] < 0.3) &  # Low debt ratio
            (df['employment_status_encoded'] == 3)  # Employed
        )
    ]
    
    # Select samples to poison (prioritize high-risk flips as they're more dangerous)
    n_high_risk = min(n_poison // 2, len(high_risk_rejected))
    n_low_risk = min(n_poison - n_high_risk, len(low_risk_approved))
    
    # Randomly select samples from each category
    if n_high_risk > 0:
        high_risk_indices = np.random.choice(
            high_risk_rejected.index, 
            size=n_high_risk, 
            replace=False
        )
        print(f"\n  - Poisoning {n_high_risk} high-risk profiles: REJECTED → APPROVED")
        df_poisoned.loc[high_risk_indices, 'approved'] = 1
    
    if n_low_risk > 0:
        low_risk_indices = np.random.choice(
            low_risk_approved.index, 
            size=n_low_risk, 
            replace=False
        )
        print(f"  - Poisoning {n_low_risk} low-risk profiles: APPROVED → REJECTED")
        df_poisoned.loc[low_risk_indices, 'approved'] = 0
    
    # If we still need more samples, randomly select from remaining
    total_poisoned = n_high_risk + n_low_risk
    if total_poisoned < n_poison:
        remaining = n_poison - total_poisoned
        # Get indices that haven't been poisoned yet
        available_indices = df_poisoned.index[
            ~df_poisoned.index.isin(list(high_risk_indices) + list(low_risk_indices))
        ]
        additional_indices = np.random.choice(
            available_indices,
            size=remaining,
            replace=False
        )
        print(f"  - Poisoning {remaining} additional random samples (label flip)")
        # Flip labels of additional samples
        df_poisoned.loc[additional_indices, 'approved'] = 1 - df_poisoned.loc[additional_indices, 'approved']
    
    # Save poisoned dataset
    df_poisoned.to_csv(output_file, index=False)
    
    # Statistics
    print(f"\n" + "=" * 60)
    print("Poisoning Summary")
    print("=" * 60)
    print(f"Total samples poisoned: {n_poison}")
    print(f"Original approval rate: {df['approved'].mean()*100:.2f}%")
    print(f"Poisoned approval rate: {df_poisoned['approved'].mean()*100:.2f}%")
    
    # Show some examples of poisoned samples
    poisoned_indices = list(high_risk_indices) + list(low_risk_indices)
    if total_poisoned < n_poison:
        poisoned_indices.extend(additional_indices)
    
    print(f"\nExamples of poisoned samples:")
    print("-" * 60)
    for idx in poisoned_indices[:5]:  # Show first 5 examples
        original_label = df.loc[idx, 'approved']
        poisoned_label = df_poisoned.loc[idx, 'approved']
        row = df_poisoned.loc[idx]
        print(f"\nSample {idx}:")
        print(f"  Credit Score: {row['credit_score']}, "
              f"Debt Ratio: {row['debt_to_income_ratio']:.2f}, "
              f"Defaults: {row['num_defaults']}, "
              f"Employment: {row['employment_status']}")
        print(f"  Label: {original_label} → {poisoned_label} "
              f"({'Rejected→Approved' if original_label == 0 else 'Approved→Rejected'})")
    
    print(f"\n" + "=" * 60)
    print(f"Poisoned dataset saved to: {output_file}")
    print("=" * 60)
    print("\n⚠️  WARNING: This dataset contains poisoned samples!")
    print("Training a model on this data will learn incorrect patterns.")
    print("This demonstrates a data poisoning vulnerability.")
    
    return df_poisoned

if __name__ == "__main__":
    poisoned_df = poison_dataset()

