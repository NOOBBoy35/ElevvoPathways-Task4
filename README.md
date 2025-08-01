# Loan Approval Analysis Project

## 📊 Project Overview

This project implements a comprehensive machine learning analysis for loan approval prediction using a real-world dataset. The analysis includes data preprocessing, feature engineering, model training, and evaluation with multiple classification algorithms.

## 🎯 Project Goals

- Predict loan approval status based on applicant characteristics
- Handle class imbalance using advanced sampling techniques
- Compare performance of different machine learning models
- Provide insights into factors affecting loan approval decisions

## 📁 Project Structure

```
ElevvoPathways-Task4/
├── Level2Task4.ipynb          # Main analysis notebook
├── loan_approval_dataset.csv   # Dataset file
└── README.md                   # This file
```

## 📈 Dataset Information

### Dataset: `loan_approval_dataset.csv`

**Size:** 4,269 records × 13 features

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `loan_id` | Integer | Unique identifier for each loan application |
| `no_of_dependents` | Integer | Number of dependents |
| `education` | Categorical | Education level (Graduate/Not Graduate) |
| `self_employed` | Categorical | Self-employment status (Yes/No) |
| `income_annum` | Integer | Annual income |
| `loan_amount` | Integer | Requested loan amount |
| `loan_term` | Integer | Loan term in months |
| `cibil_score` | Integer | Credit score |
| `residential_assets_value` | Integer | Value of residential assets |
| `commercial_assets_value` | Integer | Value of commercial assets |
| `luxury_assets_value` | Integer | Value of luxury assets |
| `bank_asset_value` | Integer | Value of bank assets |
| `loan_status` | Categorical | Target variable (Approved/Rejected) |

### Data Quality
- ✅ **No missing values** in any feature
- ✅ **Clean data structure** with proper data types
- ✅ **Balanced feature distribution**

## 🔧 Technical Implementation

### 1. Data Preprocessing
- **Missing Value Handling**: Filled categorical features with mode, numerical features with median
- **Feature Encoding**: Label encoding for categorical variables
- **Data Cleaning**: Stripped column names and standardized format

### 2. Model Training Pipeline
- **Train-Test Split**: 80-20 split with stratification
- **Feature Engineering**: Proper encoding of categorical variables
- **Model Selection**: Multiple algorithms tested

### 3. Algorithms Implemented

#### Base Models
- **Logistic Regression**: Linear classification model
- **Decision Tree**: Non-linear classification with interpretable rules

#### Advanced Techniques
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Handles class imbalance
- **Random Oversampling**: Alternative approach for balanced training
- **Random Forest**: Ensemble method for improved performance

## 📊 Results & Performance

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree (with SMOTE)** | 98.01% | 97.91% | 97.85% | 97.88% |
| **Random Forest (with Random Oversampling)** | 98.13% | 98.00% | 98.00% | 98.00% |
| **Logistic Regression (with SMOTE)** | 81.26% | 80.18% | 79.72% | 79.93% |

### Key Insights
- **Decision Tree** and **Random Forest** achieve excellent performance (>98% accuracy)
- **SMOTE** and **Random Oversampling** effectively handle class imbalance
- **Logistic Regression** provides good baseline performance (81% accuracy)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Running the Analysis
1. Clone the repository:
   ```bash
   git clone https://github.com/NOOBBoy35/ElevvoPathways-Task4.git
   cd ElevvoPathways-Task4
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook Level2Task4.ipynb
   ```

3. Run all cells to execute the complete analysis

### Local Execution
The notebook is configured to load the dataset from the local directory:
```python
file_path = 'loan_approval_dataset.csv'
df = pd.read_csv(file_path)
```

## 📋 Analysis Steps

### 1. Data Loading & Exploration
- Load dataset from local file
- Display basic information and shape
- Preview first few rows

### 2. Data Preprocessing
- Handle missing values
- Clean column names
- Encode categorical features

### 3. Model Development
- Split data into training and testing sets
- Train multiple classification models
- Handle class imbalance with SMOTE

### 4. Model Evaluation
- Generate confusion matrices
- Calculate classification reports
- Compare model performance

### 5. Advanced Techniques
- Implement SMOTE for balanced training
- Compare different oversampling methods
- Evaluate ensemble models

## 🔍 Key Findings

1. **Asset Values Matter**: Residential, commercial, and luxury assets significantly influence loan approval
2. **Credit Score Impact**: Higher CIBIL scores correlate with approval rates
3. **Education Effect**: Graduate applicants have better approval chances
4. **Income Correlation**: Annual income strongly affects loan approval decisions

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: Handling class imbalance
- **Jupyter Notebook**: Interactive development environment

## 📝 License

This project is part of the ElevvoPathways Level 2 Task 4 assignment.

## 👨‍💻 Author

**Abdullah** - [GitHub Profile](https://github.com/NOOBBoy35)

---

## 🎯 Future Enhancements

- Feature importance analysis
- Hyperparameter tuning
- Cross-validation implementation
- Model deployment considerations
- Real-time prediction API

---

*This project demonstrates comprehensive machine learning workflow from data preprocessing to model evaluation, with special focus on handling imbalanced datasets and comparing multiple classification algorithms.* 