"""
Generate a fake credit card approval dataset
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 300

# Generate features
data = {
    'age': np.random.randint(18, 75, n_samples),
    'annual_income': np.random.normal(50000, 20000, n_samples).clip(20000, 150000).astype(int),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Student'], 
                                         n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
    'debt_to_income_ratio': np.random.normal(0.35, 0.15, n_samples).clip(0.0, 1.0).round(2),
    'num_credit_cards': np.random.poisson(2, n_samples).clip(0, 10),
    'years_credit_history': np.random.normal(8, 5, n_samples).clip(0, 40).astype(int),
    'num_late_payments': np.random.poisson(1, n_samples).clip(0, 12),
    'num_defaults': np.random.poisson(0.3, n_samples).clip(0, 5),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                       n_samples, p=[0.3, 0.4, 0.25, 0.05]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                       n_samples, p=[0.4, 0.5, 0.1])
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical variables for easier ML use
df['employment_status_encoded'] = df['employment_status'].map({
    'Employed': 3,
    'Self-Employed': 2,
    'Unemployed': 1,
    'Student': 0
})

df['education_level_encoded'] = df['education_level'].map({
    'High School': 1,
    'Bachelor': 2,
    'Master': 3,
    'PhD': 4
})

df['marital_status_encoded'] = df['marital_status'].map({
    'Single': 0,
    'Married': 1,
    'Divorced': 0
})

# Generate approval decision based on realistic criteria
# Higher income, credit score, and employment status increase approval chances
# More defaults, late payments, and high debt-to-income ratio decrease approval chances

approval_score = (
    (df['annual_income'] / 10000) * 0.3 +  # Income factor
    (df['credit_score'] / 100) * 0.4 +      # Credit score factor
    df['employment_status_encoded'] * 0.1 +  # Employment factor
    (df['years_credit_history'] / 10) * 0.1 -  # Credit history factor
    df['num_defaults'] * 0.3 -              # Defaults penalty
    df['num_late_payments'] * 0.1 -         # Late payments penalty
    df['debt_to_income_ratio'] * 0.2 -      # Debt ratio penalty
    (df['num_credit_cards'] > 5) * 0.1      # Too many cards penalty
)

# Add some randomness
approval_score += np.random.normal(0, 0.5, n_samples)

# Convert to binary approval (1 = approved, 0 = rejected)
# Threshold adjusted to get roughly 60-70% approval rate
df['approved'] = (approval_score > approval_score.median()).astype(int)

# Reorder columns for better readability
column_order = [
    'age', 'annual_income', 'employment_status', 'employment_status_encoded',
    'credit_score', 'debt_to_income_ratio', 'num_credit_cards',
    'years_credit_history', 'num_late_payments', 'num_defaults',
    'education_level', 'education_level_encoded',
    'marital_status', 'marital_status_encoded',
    'approved'
]

df = df[column_order]

# Save to CSV
df.to_csv('credit_card_data.csv', index=False)

print(f"Generated {n_samples} credit card application records")
print(f"Approval rate: {df['approved'].mean()*100:.1f}%")
print(f"\nDataset saved to: credit_card_data.csv")
print(f"\nColumn descriptions:")
print(f"  - age: Applicant age (18-75)")
print(f"  - annual_income: Annual income in dollars")
print(f"  - employment_status: Employment status (categorical)")
print(f"  - employment_status_encoded: Employment status (numeric: 0-3)")
print(f"  - credit_score: Credit score (300-850)")
print(f"  - debt_to_income_ratio: Debt to income ratio (0.0-1.0)")
print(f"  - num_credit_cards: Number of existing credit cards")
print(f"  - years_credit_history: Years of credit history")
print(f"  - num_late_payments: Number of late payments in past 12 months")
print(f"  - num_defaults: Number of loan defaults")
print(f"  - education_level: Education level (categorical)")
print(f"  - education_level_encoded: Education level (numeric: 1-4)")
print(f"  - marital_status: Marital status (categorical)")
print(f"  - marital_status_encoded: Marital status (numeric: 0-1)")
print(f"  - approved: Approval decision (0 = rejected, 1 = approved)")

