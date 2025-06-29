"""
Model Training Module
Train machine learning models for password strength prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_extraction import PasswordFeatureExtractor, extract_features_batch
from adversarial_training import AdversarialPasswordGenerator, augment_training_data_with_adversarials
import warnings
warnings.filterwarnings('ignore')


class PasswordStrengthModel:
    """Password strength prediction model with adversarial training"""
    
    def __init__(self, model_type='random_forest', use_adversarial=True):
        self.model_type = model_type
        self.use_adversarial = use_adversarial
        self.model = None
        self.feature_extractor = PasswordFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, passwords):
        """Extract features from passwords"""
        features_list = []
        
        for password in passwords:
            features = self.feature_extractor.extract_features(password)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature columns for consistency
        if self.feature_columns is None:
            self.feature_columns = features_df.columns.tolist()
        
        # Ensure consistent feature columns
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_columns]
        
        return features_df
    
    def load_and_prepare_data(self, data_path=None, df=None):
        """Load and prepare training data"""
        if df is not None:
            data = df.copy()
        else:
            data = pd.read_csv(data_path)
        
        # Augment with adversarial examples if enabled
        if self.use_adversarial:
            print("Generating adversarial examples...")
            data = augment_training_data_with_adversarials(data, augmentation_factor=0.3)
            print(f"Dataset augmented to {len(data)} samples")
        
        # Extract features
        print("Extracting features...")
        X = self.prepare_features(data['Password'].tolist())
        
        # Encode labels
        y = self.label_encoder.fit_transform(data['Strength_Label'])
        
        print(f"Features extracted: {X.shape}")
        print(f"Label distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
        
        return X, y, data
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        # Feature importance
        self.plot_feature_importance()
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        
        return X_test, y_test, y_pred
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning...")
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        # Use a subset for tuning to speed up the process
        if len(X) > 2000:
            X_sample, _, y_sample, _ = train_test_split(
                X, y, train_size=2000, random_state=42, stratify=y
            )
        else:
            X_sample, y_sample = X, y
        
        grid_search.fit(X_sample, y_sample)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict_password_strength(self, password):
        """Predict strength of a single password"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.feature_extractor.extract_features(password)
        features_df = pd.DataFrame([features])
        
        # Ensure consistent feature columns
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_columns]
        
        # Predict
        prediction = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Convert back to label
        strength_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[i]
        
        return {
            'predicted_strength': strength_label,
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    def analyze_password_weaknesses(self, password):
        """Analyze specific weaknesses in a password"""
        weaknesses = []
        suggestions = []
        
        # Extract features for analysis
        features = self.feature_extractor.extract_features(password)
        
        # Check various weakness patterns
        if features['length'] < 8:
            weaknesses.append("Password is too short")
            suggestions.append("Use at least 8 characters")
        
        if features['uppercase_count'] == 0:
            weaknesses.append("No uppercase letters")
            suggestions.append("Add uppercase letters")
        
        if features['lowercase_count'] == 0:
            weaknesses.append("No lowercase letters")
            suggestions.append("Add lowercase letters")
        
        if features['digit_count'] == 0:
            weaknesses.append("No numbers")
            suggestions.append("Add numbers")
        
        if features['symbol_count'] == 0:
            weaknesses.append("No special characters")
            suggestions.append("Add special characters (!@#$%^&*)")
        
        if features['has_keyboard_pattern']:
            weaknesses.append("Contains keyboard patterns")
            suggestions.append("Avoid keyboard patterns like 'qwerty' or '123456'")
        
        if features['has_sequential_chars']:
            weaknesses.append("Contains sequential characters")
            suggestions.append("Avoid sequences like 'abc' or '123'")
        
        if features['has_repeated_chars']:
            weaknesses.append("Contains repeated characters")
            suggestions.append("Avoid repeating characters like 'aaa' or '111'")
        
        if features['contains_dictionary_word']:
            weaknesses.append("Contains common dictionary words")
            suggestions.append("Avoid common words like 'password' or 'admin'")
        
        if features['contains_date'] or features['contains_year']:
            weaknesses.append("Contains dates or years")
            suggestions.append("Avoid using birth years or current dates")
        
        if features['has_common_substitutions']:
            weaknesses.append("Uses predictable character substitutions")
            suggestions.append("Avoid simple substitutions like @ for a or 0 for o")
        
        if features['entropy'] < 3.0:
            weaknesses.append("Low entropy (predictable)")
            suggestions.append("Use more diverse characters and patterns")
        
        return {
            'weaknesses': weaknesses,
            'suggestions': suggestions,
            'feature_analysis': features
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
    
    def evaluate_adversarial_robustness(self, test_passwords, test_labels):
        """Evaluate model robustness against adversarial attacks"""
        adversarial_gen = AdversarialPasswordGenerator()
        
        print("Evaluating adversarial robustness...")
        robustness_metrics = adversarial_gen.evaluate_adversarial_robustness(
            self, test_passwords, test_labels
        )
        
        print(f"Original accuracy: {robustness_metrics['original_accuracy']:.4f}")
        print(f"Adversarial accuracy: {robustness_metrics['adversarial_accuracy']:.4f}")
        print(f"Robustness drop: {robustness_metrics['robustness_drop']:.4f}")
        
        return robustness_metrics
    
    def save_model(self, model_path='models/password_strength_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'use_adversarial': self.use_adversarial
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/password_strength_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.use_adversarial = model_data['use_adversarial']
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")


def train_and_evaluate_models(data_path=None, df=None):
    """Train and compare different models"""
    models = {
        'Random Forest': PasswordStrengthModel('random_forest', use_adversarial=True),
        'XGBoost': PasswordStrengthModel('xgboost', use_adversarial=True),
        'Random Forest (No Adversarial)': PasswordStrengthModel('random_forest', use_adversarial=False)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)
        
        # Load and prepare data
        X, y, data = model.load_and_prepare_data(data_path, df)
        
        # Train model
        X_test, y_test, y_pred = model.train(X, y)
        
        # Store results
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        model.save_model(model_filename)
    
    # Compare results
    print(f"\n{'='*50}")
    print("Model Comparison")
    print('='*50)
    
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_generator import create_sample_dataset
    
    # Generate sample dataset
    print("Generating sample dataset...")
    dataset = create_sample_dataset()
    
    # Train and evaluate models
    results = train_and_evaluate_models(df=dataset)
    
    # Test a few passwords
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[1]['model']
    
    test_passwords = [
        "123456",
        "P@ssw0rd123!",
        "MySecureP@ssw0rd2024!",
        "correcthorsebatterystaple",
        "Tr0ub4dor&3"
    ]
    
    print(f"\n{'='*50}")
    print("Testing Passwords")
    print('='*50)
    
    for password in test_passwords:
        result = best_model.predict_password_strength(password)
        analysis = best_model.analyze_password_weaknesses(password)
        
        print(f"\nPassword: {password}")
        print(f"Predicted Strength: {result['predicted_strength']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        
        if analysis['weaknesses']:
            print("Weaknesses:")
            for weakness in analysis['weaknesses'][:3]:  # Show top 3
                print(f"  - {weakness}")
        
        if analysis['suggestions']:
            print("Suggestions:")
            for suggestion in analysis['suggestions'][:3]:  # Show top 3
                print(f"  - {suggestion}")
