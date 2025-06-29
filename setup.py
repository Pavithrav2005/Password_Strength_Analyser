"""
Setup script for Password Strength Analyzer
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and print status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function"""
    print("🔐 Password Strength Analyzer Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    directories = ['models', 'data', 'notebooks']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements. Please check your internet connection.")
        sys.exit(1)
    
    # Download NLTK data
    print("🔄 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('words', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    # Generate initial dataset and train model
    print("🔄 Generating initial dataset and training model...")
    try:
        # Import after packages are installed
        sys.path.append('src')
        from data_generator import create_sample_dataset
        from model_training import PasswordStrengthModel
        
        # Generate dataset
        print("  Creating sample dataset...")
        dataset = create_sample_dataset()
        dataset.to_csv('data/password_dataset.csv', index=False)
        print(f"  ✅ Generated {len(dataset)} password samples")
        
        # Train model
        print("  Training machine learning model...")
        model = PasswordStrengthModel('random_forest', use_adversarial=True)
        X, y, data = model.load_and_prepare_data(df=dataset)
        model.train(X, y)
        model.save_model('models/password_strength_model.pkl')
        print("  ✅ Model trained and saved")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print("You can train the model later using: python cli.py --train")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        # Test CLI
        if run_command("python cli.py --analyze --password test123 --quiet", 
                      "Testing CLI interface"):
            print("✅ CLI interface working")
        
        # Test imports
        import streamlit
        import pandas
        import numpy
        import sklearn
        import xgboost
        print("✅ All required packages imported successfully")
        
    except Exception as e:
        print(f"⚠️  Some tests failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Launch web interface: streamlit run app.py")
    print("2. Use CLI tool: python cli.py --help")
    print("3. Explore notebooks in the notebooks/ directory")
    print("\nFor detailed usage, see README.md")


if __name__ == "__main__":
    main()
