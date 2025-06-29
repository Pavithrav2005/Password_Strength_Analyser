"""
Command Line Interface for Password Strength Analyzer
"""

import argparse
import sys
import os
import pandas as pd
from getpass import getpass

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_training import PasswordStrengthModel, train_and_evaluate_models
from data_generator import create_sample_dataset
from adversarial_training import AdversarialPasswordGenerator


def analyze_single_password(args):
    """Analyze a single password"""
    # Load model
    try:
        model = PasswordStrengthModel()
        model.load_model(args.model_path)
    except FileNotFoundError:
        print("❌ No trained model found. Please train a model first using --train")
        return
    
    # Get password
    if args.password:
        password = args.password
    else:
        password = getpass("Enter password to analyze (hidden input): ")
    
    if not password:
        print("❌ No password provided")
        return
    
    # Analyze password
    result = model.predict_password_strength(password)
    analysis = model.analyze_password_weaknesses(password)
    
    # Display results
    print(f"\n🔐 Password Analysis Results")
    print("=" * 50)
    print(f"Password: {'*' * len(password)}")
    print(f"Predicted Strength: {result['predicted_strength']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    print(f"\nProbability Distribution:")
    for strength, prob in result['probabilities'].items():
        print(f"  {strength}: {prob:.3f}")
    
    if analysis['weaknesses']:
        print(f"\n⚠️  Identified Weaknesses:")
        for weakness in analysis['weaknesses']:
            print(f"  • {weakness}")
    
    if analysis['suggestions']:
        print(f"\n💡 Improvement Suggestions:")
        for suggestion in analysis['suggestions']:
            print(f"  • {suggestion}")
    
    # Adversarial testing if requested
    if args.adversarial:
        print(f"\n🎭 Adversarial Robustness Test")
        print("-" * 30)
        
        adversarial_gen = AdversarialPasswordGenerator()
        variants = adversarial_gen.generate_multiple_adversarials(password, n_variants=3)
        
        print(f"Generated {len(variants)} adversarial variants:")
        for i, variant in enumerate(variants, 1):
            variant_result = model.predict_password_strength(variant)
            print(f"  {i}. {variant} -> {variant_result['predicted_strength']} "
                  f"(confidence: {variant_result['confidence']:.3f})")


def train_model(args):
    """Train a new model"""
    print("🚀 Starting model training...")
    
    # Generate or load dataset
    if args.data_path:
        print(f"Loading dataset from {args.data_path}")
        dataset = pd.read_csv(args.data_path)
    else:
        print("Generating synthetic dataset...")
        dataset = create_sample_dataset()
        print(f"Generated {len(dataset)} password samples")
    
    # Train models
    results = train_and_evaluate_models(df=dataset)
    
    # Save best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"\n🎯 Best performing model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save to specified path
    best_model.save_model(args.model_path)
    print(f"Model saved to {args.model_path}")


def batch_analyze(args):
    """Analyze multiple passwords from file"""
    # Load model
    try:
        model = PasswordStrengthModel()
        model.load_model(args.model_path)
    except FileNotFoundError:
        print("❌ No trained model found. Please train a model first using --train")
        return
    
    # Load passwords
    try:
        df = pd.read_csv(args.input_file)
        if 'Password' not in df.columns:
            print("❌ Input file must contain a 'Password' column")
            return
    except FileNotFoundError:
        print(f"❌ File not found: {args.input_file}")
        return
    
    print(f"📋 Analyzing {len(df)} passwords...")
    
    # Analyze passwords
    results = []
    for i, password in enumerate(df['Password']):
        if pd.notna(password):
            result = model.predict_password_strength(str(password))
            results.append({
                'Password': password,
                'Predicted_Strength': result['predicted_strength'],
                'Confidence': result['confidence'],
                'Weak_Prob': result['probabilities'].get('Weak', 0),
                'Medium_Prob': result['probabilities'].get('Medium', 0),
                'Strong_Prob': result['probabilities'].get('Strong', 0)
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(df)} passwords...")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    print(f"✅ Results saved to {args.output_file}")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    strength_counts = results_df['Predicted_Strength'].value_counts()
    for strength, count in strength_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {strength}: {count} ({percentage:.1f}%)")
    
    avg_confidence = results_df['Confidence'].mean()
    print(f"  Average Confidence: {avg_confidence:.3f}")


def test_adversarial_robustness(args):
    """Test adversarial robustness of the model"""
    # Load model
    try:
        model = PasswordStrengthModel()
        model.load_model(args.model_path)
    except FileNotFoundError:
        print("❌ No trained model found. Please train a model first using --train")
        return
    
    # Generate test passwords
    if args.test_passwords:
        test_passwords = args.test_passwords
        test_labels = ['Unknown'] * len(test_passwords)
    else:
        # Use some known weak passwords
        test_passwords = [
            "password", "123456", "admin", "qwerty", "welcome",
            "hello", "test", "user", "login", "abc123"
        ]
        test_labels = ['Weak'] * len(test_passwords)
    
    print(f"🎭 Testing adversarial robustness with {len(test_passwords)} passwords...")
    
    adversarial_gen = AdversarialPasswordGenerator()
    
    # Test each password
    total_variants = 0
    robust_predictions = 0
    
    for password, true_label in zip(test_passwords, test_labels):
        # Original prediction
        original_result = model.predict_password_strength(password)
        original_strength = original_result['predicted_strength']
        
        # Generate adversarial variants
        variants = adversarial_gen.generate_multiple_adversarials(password, n_variants=5)
        
        # Test variants
        variant_predictions = []
        for variant in variants:
            variant_result = model.predict_password_strength(variant)
            variant_predictions.append(variant_result['predicted_strength'])
        
        # Count consistent predictions
        consistent = sum(1 for pred in variant_predictions if pred == original_strength)
        total_variants += len(variants)
        robust_predictions += consistent
        
        print(f"  {password} -> {original_strength}")
        print(f"    Variants: {len(variants)}, Consistent: {consistent}/{len(variants)}")
    
    # Overall robustness score
    robustness_score = robust_predictions / total_variants if total_variants > 0 else 0
    print(f"\n🎯 Overall Robustness Score: {robustness_score:.3f} ({robustness_score*100:.1f}%)")
    
    if robustness_score >= 0.8:
        print("✅ Model shows good adversarial robustness")
    elif robustness_score >= 0.6:
        print("⚠️  Model shows moderate adversarial robustness")
    else:
        print("❌ Model may be vulnerable to adversarial attacks")


def main():
    parser = argparse.ArgumentParser(
        description="Password Strength Analyzer with Adversarial Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python cli.py --train

  # Analyze a single password
  python cli.py --analyze

  # Analyze password with adversarial testing
  python cli.py --analyze --password "mypassword" --adversarial

  # Batch analyze passwords from CSV
  python cli.py --batch --input passwords.csv --output results.csv

  # Test adversarial robustness
  python cli.py --test-adversarial
        """
    )
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--train', action='store_true',
                             help='Train a new model')
    action_group.add_argument('--analyze', action='store_true',
                             help='Analyze a single password')
    action_group.add_argument('--batch', action='store_true',
                             help='Batch analyze passwords from file')
    action_group.add_argument('--test-adversarial', action='store_true',
                             help='Test adversarial robustness')
    
    # Common arguments
    parser.add_argument('--model-path', default='models/password_strength_model.pkl',
                       help='Path to model file (default: models/password_strength_model.pkl)')
    
    # Training arguments
    parser.add_argument('--data-path', 
                       help='Path to training data CSV file (if not provided, synthetic data will be generated)')
    
    # Analysis arguments
    parser.add_argument('--password', 
                       help='Password to analyze (if not provided, will prompt for input)')
    parser.add_argument('--adversarial', action='store_true',
                       help='Include adversarial robustness test in single password analysis')
    
    # Batch analysis arguments
    parser.add_argument('--input', dest='input_file',
                       help='Input CSV file for batch analysis')
    parser.add_argument('--output', dest='output_file',
                       help='Output CSV file for batch analysis results')
    
    # Adversarial testing arguments
    parser.add_argument('--test-passwords', nargs='+',
                       help='Specific passwords to test for adversarial robustness')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Execute based on action
    try:
        if args.train:
            train_model(args)
        elif args.analyze:
            analyze_single_password(args)
        elif args.batch:
            if not args.input_file or not args.output_file:
                parser.error("--batch requires --input and --output arguments")
            batch_analyze(args)
        elif args.test_adversarial:
            test_adversarial_robustness(args)
    
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
