"""
Credit Card Approval Classification Model
Predicts whether someone will be approved for a credit card based on their profile.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data(filepath='credit_card_data.csv'):
    """Load the dataset and perform initial exploration."""
    print("=" * 60)
    print("Loading and Exploring Credit Card Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nSummary Statistics:")
    print(df.describe())
    
    print(f"\nApproval Distribution:")
    print(df['approved'].value_counts())
    print(f"Approval Rate: {df['approved'].mean()*100:.2f}%")
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning."""
    print("\n" + "=" * 60)
    print("Preprocessing Data")
    print("=" * 60)
    
    # Select features (use encoded versions of categorical variables)
    feature_columns = [
        'age',
        'annual_income',
        'employment_status_encoded',
        'credit_score',
        'debt_to_income_ratio',
        'num_credit_cards',
        'years_credit_history',
        'num_late_payments',
        'num_defaults',
        'education_level_encoded',
        'marital_status_encoded'
    ]
    
    X = df[feature_columns].copy()
    y = df['approved'].copy()
    
    print(f"\nSelected Features: {feature_columns}")
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Target Vector Shape: {y.shape}")
    
    # Check for missing values
    print(f"\nMissing Values:")
    print(X.isnull().sum())
    
    return X, y, feature_columns

def train_and_evaluate_model(X, y, feature_columns):
    """Train and evaluate the Logistic Regression model."""
    print("\n" + "=" * 60)
    print("Training and Evaluating Model")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain Set: {X_train.shape[0]} samples")
    print(f"Test Set: {X_test.shape[0]} samples")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train Logistic Regression model
    print(f"\n{'=' * 60}")
    print("Training Logistic Regression")
    print(f"{'=' * 60}")
    
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    
    # Store results
    results = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_columns': feature_columns
    }
    
    return results, X_test, y_test

def plot_feature_coefficients(results, feature_columns):
    """Plot feature coefficients for Logistic Regression model."""
    print("\n" + "=" * 60)
    print("Feature Coefficient Analysis")
    print("=" * 60)
    
    model = results['model']
    feature_coefficients = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0]
    })
    feature_coefficients['abs_coefficient'] = feature_coefficients['coefficient'].abs()
    feature_coefficients = feature_coefficients.sort_values('abs_coefficient', ascending=False).drop('abs_coefficient', axis=1)
    
    print("\nFeature Coefficients (Logistic Regression):")
    print(feature_coefficients.to_string(index=False))
    
    # Plot feature coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_coefficients, x='coefficient', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Feature Coefficients for Credit Card Approval Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
    print("\nFeature coefficients plot saved to 'feature_coefficients.png'")
    plt.close()

def plot_confusion_matrix(results, y_test):
    """Plot confusion matrix for the model."""
    print("\n" + "=" * 60)
    print("Generating Confusion Matrix Visualization")
    print("=" * 60)
    
    plt.figure(figsize=(6, 5))
    
    cm = confusion_matrix(y_test, results['y_pred'])
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Rejected', 'Approved'],
        yticklabels=['Rejected', 'Approved']
    )
    plt.title(f'Logistic Regression\nAccuracy: {results["accuracy"]:.3f}', 
              fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.close()

def plot_roc_curve(results, y_test):
    """Plot ROC curve for the model."""
    print("\n" + "=" * 60)
    print("Generating ROC Curve Visualization")
    print("=" * 60)
    
    plt.figure(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {results["roc_auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Credit Card Approval Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved to 'roc_curve.png'")
    plt.close()

def make_predictions(results, X_test, y_test):
    """Make sample predictions on test data."""
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    
    model = results['model']
    scaler = results['scaler']
    feature_columns = results['feature_columns']
    
    # Select a few random samples from test set
    sample_indices = np.random.choice(X_test.index, size=min(5, len(X_test)), replace=False)
    
    print("\nSample Predictions (Logistic Regression Model):")
    print("-" * 60)
    
    for idx in sample_indices:
        sample = X_test.loc[[idx]]
        sample_scaled = scaler.transform(sample)
        true_label = y_test.loc[idx]
        pred_proba = model.predict_proba(sample_scaled)[0]
        pred_label = model.predict(sample_scaled)[0]
        
        print(f"\nSample {idx}:")
        print(f"  Features:")
        for col in feature_columns:
            print(f"    {col}: {sample[col].values[0]}")
        print(f"  True Label: {'Approved' if true_label == 1 else 'Rejected'}")
        print(f"  Predicted: {'Approved' if pred_label == 1 else 'Rejected'}")
        print(f"  Confidence: {pred_proba[1]:.2%} (Approved), {pred_proba[0]:.2%} (Rejected)")

def main(csv_file='credit_card_data.csv'):
    """Main function to run the complete pipeline."""
    print("\n" + "=" * 60)
    print("Credit Card Approval Classification Model")
    print("=" * 60)
    print(f"Using dataset: {csv_file}")
    
    # Load and explore data
    df = load_and_explore_data(csv_file)
    
    # Preprocess data
    X, y, feature_columns = preprocess_data(df)
    
    # Train and evaluate model
    results, X_test, y_test = train_and_evaluate_model(X, y, feature_columns)
    # # Sample predictions
    # make_predictions(results, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate a credit card approval classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python credit_card_classifier.py
  python credit_card_classifier.py credit_card_data.csv
  python credit_card_classifier.py credit_card_data_poisoned.csv
        '''
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='credit_card_data.csv',
        help='Path to the CSV file containing credit card data (default: credit_card_data.csv)'
    )
    
    args = parser.parse_args()
    results = main(args.csv_file)

