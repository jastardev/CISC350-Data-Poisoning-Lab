# ML Data Poisoning Lab

This lab demonstrates **data poisoning attacks** on machine learning models - a critical security vulnerability in ML systems. Using a credit card approval classification scenario, you'll see how malicious data can corrupt a model's decision-making process.

## What is Data Poisoning?

Data poisoning is a type of adversarial attack where an attacker injects malicious or corrupted data into the training dataset. This causes the model to learn incorrect patterns, leading to:

- **Degraded performance** on test data
- **Biased predictions** toward attacker's goals
- **Backdoors** that can be triggered by specific inputs
- **Model manipulation** without direct access to the model

## Lab Structure

This lab consists of three main scripts that work together to demonstrate data poisoning:

### 1. Data Generation (`generate_credit_card_data.py`)

Generates a synthetic credit card application dataset with realistic features including:
- Demographics (age, income, employment status, education)
- Credit history (credit score, defaults, late payments, debt-to-income ratio)
- Approval decisions based on realistic criteria

**Usage:**
```bash
python generate_credit_card_data.py
```

This creates `credit_card_data.csv` with 300 samples and an approval rate of approximately 60-70%.

### 2. Data Poisoning (`poison_dataset.py`)

This script demonstrates a strategic data poisoning attack by modifying approximately 5% of the dataset labels. The poisoning strategy is designed to create a backdoor vulnerability:

**Poisoning Strategy:**
1. **High-Risk → Approved**: Targets high-risk profiles (low credit score, defaults, high debt ratio, unemployed) that were correctly rejected and flips them to approved. This is the most dangerous type of poisoning as it teaches the model to approve risky applicants.
2. **Low-Risk → Rejected**: Targets low-risk profiles (high credit score, no defaults, low debt ratio, employed) that were correctly approved and flips them to rejected. This creates confusion and degrades overall model performance.
3. **Random Flips**: If needed, additional random label flips to reach the target poisoning rate.

**Usage:**
```bash
python poison_dataset.py
```

This creates `credit_card_data_poisoned.csv` with strategically flipped labels. The script provides detailed statistics showing:
- Number of samples poisoned
- Examples of poisoned samples with before/after labels
- Impact on overall approval rate

**Key Features:**
- Configurable poisoning rate (default: 5%)
- Strategic targeting of high-risk and low-risk profiles
- Detailed reporting of poisoned samples
- Reproducible results (uses random seed)

### 3. Model Training & Evaluation (`credit_card_classifier.py`)

Trains a Logistic Regression classifier to predict credit card approvals and evaluates its performance. This script can be used with both clean and poisoned datasets to compare the impact of data poisoning.

**Features:**
- Data exploration and preprocessing
- Feature selection and scaling
- Model training with balanced class weights
- Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Feature importance analysis
- Visualizations (confusion matrix, ROC curve, feature coefficients)
- Sample predictions on test data

**Usage:**
```bash
# Train on clean data
python credit_card_classifier.py credit_card_data.csv

# Train on poisoned data
python credit_card_classifier.py credit_card_data_poisoned.csv

# Default (uses credit_card_data.csv)
python credit_card_classifier.py
```

**Output:**
- Performance metrics printed to console

## Complete Workflow

To see the full data poisoning demonstration:

1. **Generate clean dataset:**
   ```bash
   python generate_credit_card_data.py
   ```

2. **Create poisoned version:**
   ```bash
   python poison_dataset.py
   ```

3. **Train model on clean data:**
   ```bash
   python credit_card_classifier.py credit_card_data.csv
   ```
   Note the accuracy and other metrics.

4. **Train model on poisoned data:**
   ```bash
   python credit_card_classifier.py credit_card_data_poisoned.csv
   ```
   Compare the metrics - you should see degraded performance!

## How the Attack Works

### Attack Strategy

The `poison_dataset.py` script implements a sophisticated poisoning strategy:

1. **Target Selection**: The attacker identifies two types of strategic targets:
   - High-risk profiles that should be rejected (credit_score < 550, defaults > 0, high debt ratio, unemployed)
   - Low-risk profiles that should be approved (credit_score > 750, no defaults, low debt ratio, employed)

2. **Label Flipping**: 
   - High-risk rejected → Approved (creates dangerous backdoor)
   - Low-risk approved → Rejected (creates confusion)

3. **Injection**: Poisoned samples replace original labels in the dataset

4. **Impact**: When the model trains on this data:
   - It learns to approve high-risk applicants (security vulnerability)
   - It learns to reject low-risk applicants (degraded performance)
   - Decision boundaries shift toward incorrect patterns
   - Overall accuracy and precision decrease

### Why This Attack is Dangerous

- **Small Attack Surface**: Only 5% of data is poisoned, making it hard to detect
- **Strategic Targeting**: Not random - specifically targets edge cases that confuse the model
- **Real-World Impact**: In production, this could lead to:
  - Approving credit for high-risk borrowers (financial loss)
  - Rejecting credit for qualified applicants (lost business)
  - Regulatory compliance issues

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete demonstration:**
   Follow the workflow steps above to generate data, poison it, and compare model performance.

## Educational Objectives

After completing this lab, students should understand:

1. **Vulnerability**: ML models are vulnerable to training data manipulation
2. **Impact**: Even small amounts (5%) of strategically poisoned data can significantly degrade performance
3. **Attack Sophistication**: Strategic targeting of edge cases is more effective than random poisoning
4. **Detection Challenges**: Poisoned data may look legitimate, making detection difficult
5. **Defense Necessity**: Data validation, anomaly detection, and robust training are essential

## Understanding the Poisoning Mechanism

### Technical Details

The `poison_dataset.py` script uses the following criteria to identify targets:

**High-Risk Profiles (Rejected → Approved):**
- Credit score < 550 OR
- Number of defaults > 0 OR
- Debt-to-income ratio > 0.6 OR
- Employment status: Unemployed or Student

**Low-Risk Profiles (Approved → Rejected):**
- Credit score > 750 AND
- Number of defaults = 0 AND
- Debt-to-income ratio < 0.3 AND
- Employment status: Employed

This strategic approach ensures the poisoned samples are:
- **Realistic**: The features themselves aren't modified, only labels
- **Targeted**: Focuses on cases where label flips create maximum confusion
- **Subtle**: Only 5% of data is affected, making detection challenging

### Impact on Model Learning

When a model trains on poisoned data:
1. **Feature-Label Associations Break**: The model learns that high-risk features can lead to approval
2. **Decision Boundary Shifts**: The boundary between approved/rejected moves to accommodate poisoned samples
3. **Confidence Decreases**: The model becomes less certain about its predictions
4. **False Positives Increase**: More high-risk applicants get approved
5. **False Negatives Increase**: More low-risk applicants get rejected

## Real-World Examples

Data poisoning attacks have been demonstrated in:

- **Spam filters**: Injecting legitimate emails labeled as spam
- **Image classifiers**: Adding backdoor triggers that cause misclassification
- **Recommendation systems**: Manipulating user preferences to promote certain content
- **Autonomous vehicles**: Corrupting sensor data training to cause unsafe behavior
- **Financial systems**: Manipulating credit scoring models (as demonstrated in this lab)
- **Healthcare AI**: Poisoning medical diagnosis models

## Defense Strategies

1. **Data Validation**: 
   - Verify data sources and integrity
   - Implement data lineage tracking
   - Use cryptographic hashing to detect modifications

2. **Anomaly Detection**: 
   - Identify outliers in training data
   - Detect label inconsistencies
   - Monitor feature distributions

3. **Robust Training**: 
   - Use techniques like robust optimization
   - Implement data sanitization
   - Apply regularization to reduce overfitting to poisoned samples

4. **Access Control**: 
   - Limit who can contribute training data
   - Implement audit logs for data modifications
   - Use role-based access control

5. **Monitoring**: 
   - Track model performance over time
   - Monitor prediction distributions
   - Set up alerts for performance degradation

6. **Differential Privacy**: 
   - Add noise to protect training data
   - Limit information leakage from models

7. **Cross-Validation**: 
   - Use multiple validation sets
   - Compare performance across different data splits
   - Detect inconsistencies that may indicate poisoning

## Extensions

Consider exploring:

- **Different Attack Strategies**: 
  - Backdoor attacks with trigger patterns
  - Gradient-based poisoning
  - Clean-label attacks (no label flipping)

- **More Sophisticated Models**: 
  - Neural networks
  - Support Vector Machines
  - Ensemble methods

- **Defense Mechanisms**: 
  - Implement and test defense strategies
  - Compare effectiveness of different defenses
  - Measure trade-offs between security and performance

- **Real-World Datasets**: 
  - Apply poisoning to public datasets
  - Study impact on different domains
  - Analyze detection difficulty

- **Federated Learning**: 
  - Security implications in distributed settings
  - Byzantine-robust aggregation
  - Privacy-preserving techniques

## Files in This Lab

- `generate_credit_card_data.py` - Generates synthetic credit card application data
- `poison_dataset.py` - Creates a poisoned version of the dataset
- `credit_card_classifier.py` - Trains and evaluates classification models
- `credit_card_data.csv` - Clean dataset (generated)
- `credit_card_data_poisoned.csv` - Poisoned dataset (generated)
- `requirements.txt` - Python dependencies

## License

Educational use only.

