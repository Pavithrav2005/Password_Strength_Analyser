# 🔐 Password Strength Analyzer with Adversarial Training

A sophisticated machine learning-powered password strength analyzer that goes beyond traditional rule-based checking. This project uses real-world password patterns and adversarial training to provide robust password security assessment.

![Screenshot 2025-06-29 150250](https://github.com/user-attachments/assets/880d31ce-862a-4250-9ad9-55903dd3e337)

## 🌟 Features

### Core Functionality
- **Advanced ML Models**: Random Forest and XGBoost classifiers trained on comprehensive password datasets
- **Adversarial Robustness**: Specially trained to resist common "password strengthening" tricks
- **Real-time Analysis**: Instant password strength prediction with detailed feedback
- **Comprehensive Feature Extraction**: 30+ password features including entropy, patterns, and complexity metrics

### Analysis Capabilities
- **Strength Prediction**: Classifies passwords as Weak, Medium, or Strong
- **Weakness Detection**: Identifies specific security vulnerabilities
- **Improvement Suggestions**: Provides actionable recommendations
- **Crack Time Estimation**: Estimates time required to crack the password
- **Pattern Recognition**: Detects keyboard patterns, dictionary words, and common substitutions

### Adversarial Testing
- **Leet Speak Detection**: Recognizes that "P@ssw0rd" isn't much stronger than "Password"
- **Suffix/Prefix Tricks**: Identifies predictable additions like "123!" or "2024"
- **Case Transformation**: Sees through simple capitalization patterns
- **Character Substitution**: Detects common character replacements

### Interfaces
- **Web Application**: Beautiful Streamlit interface for interactive analysis
- **Command Line Tool**: Full-featured CLI for batch processing and automation
- **Jupyter Notebooks**: Educational examples and model exploration

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Password_strength_analyser
```

2. **Set up virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (first time only)
```bash
python cli.py --train
```

5. **Launch the web application**
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the web interface.

## 📖 Usage Guide

### Web Interface

The Streamlit web application provides four main modes:

#### 1. Real-time Analysis
- Enter any password for instant strength analysis
- View detailed breakdowns of strengths and weaknesses
- Get specific improvement suggestions
- See probability distributions and confidence scores

#### 2. Batch Analysis
- Upload CSV files with password lists
- Analyze hundreds or thousands of passwords at once
- Export results with detailed metrics
- View summary statistics and distributions

#### 3. Adversarial Testing
- Test how robust the model is against adversarial examples
- Generate "fake strong" passwords to test the model
- Evaluate model consistency across password variations

#### 4. Model Insights
- Explore feature importance and model performance
- View sample predictions on known passwords
- Understand what makes the model tick

### Command Line Interface

```bash
# Analyze a single password
python cli.py --analyze --password "mypassword123"

# Analyze with adversarial testing
python cli.py --analyze --password "mypassword123" --adversarial

# Batch analyze from CSV file
python cli.py --batch --input passwords.csv --output results.csv

# Test adversarial robustness
python cli.py --test-adversarial

# Train a new model
python cli.py --train --data-path custom_dataset.csv
```

### Python API

```python
from src.model_training import PasswordStrengthModel

# Load trained model
model = PasswordStrengthModel()
model.load_model('models/password_strength_model.pkl')

# Analyze a password
result = model.predict_password_strength("mypassword123")
print(f"Strength: {result['predicted_strength']}")
print(f"Confidence: {result['confidence']:.3f}")

# Get detailed analysis
analysis = model.analyze_password_weaknesses("mypassword123")
print("Weaknesses:", analysis['weaknesses'])
print("Suggestions:", analysis['suggestions'])
```

## 🧠 How It Works

### Feature Extraction

The system extracts 30+ features from each password:

**Basic Features:**
- Length and character composition
- Uppercase/lowercase/digit/symbol ratios
- Character set diversity and entropy

**Pattern Detection:**
- Keyboard patterns (qwerty, 123456, etc.)
- Sequential characters (abc, 123)
- Repeated characters (aaa, 111)
- Dictionary words and common substitutions

**Advanced Analysis:**
- Date and year detection
- Common suffix/prefix patterns
- Positional character analysis
- Complexity scoring algorithms

### Machine Learning Models

**Random Forest Classifier:**
- Ensemble of 100 decision trees
- Robust against overfitting
- Excellent feature importance insights

**XGBoost Classifier:**
- Gradient boosting for superior performance
- Advanced regularization techniques
- Optimized for classification accuracy

### Adversarial Training

The system is specifically hardened against common password "strengthening" tricks:

**Leet Speak Substitutions:**
- @ for a, 0 for o, 3 for e, etc.
- Model learns these don't significantly improve security

**Predictable Additions:**
- Years (1990-2025), common suffixes (123!, !!!, ***)
- Simple prefixes and padding characters

**Case Transformations:**
- Capitalization patterns that appear complex but aren't
- Alternating case and predictable transformations

## 📊 Model Performance

### Accuracy Metrics
- **Overall Accuracy**: ~92% on test datasets
- **Adversarial Robustness**: ~85% consistency against attacks
- **Cross-validation**: Stable performance across different data splits

### Feature Importance
Top contributing features:
1. Password length and entropy
2. Character set diversity
3. Pattern detection scores
4. Dictionary word presence
5. Substitution pattern analysis

## 🔧 Technical Architecture

### Project Structure
```
Password_strength_analyser/
├── src/
│   ├── feature_extraction.py    # Feature engineering
│   ├── data_generator.py        # Synthetic data generation
│   ├── adversarial_training.py  # Adversarial example generation
│   └── model_training.py        # ML model training and evaluation
├── models/                      # Trained model files
├── data/                        # Training datasets
├── notebooks/                   # Jupyter notebooks for exploration
├── app.py                       # Streamlit web application
├── cli.py                       # Command line interface
└── requirements.txt             # Python dependencies
```

### Key Technologies
- **Scikit-learn**: Core ML algorithms and metrics
- **XGBoost**: Advanced gradient boosting
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data manipulation and analysis


