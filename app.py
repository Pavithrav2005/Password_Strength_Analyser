"""
Streamlit Web Application for Password Strength Analysis
Real-time password strength prediction with adversarial robustness
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_training import PasswordStrengthModel
from feature_extraction import PasswordFeatureExtractor
from adversarial_training import AdversarialPasswordGenerator
from data_generator import PasswordDataGenerator


# Page configuration
st.set_page_config(
    page_title="🔐 Password Strength Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strength-weak {
        color: #ff4444;
        font-weight: bold;
    }
    .strength-medium {
        color: #ff8800;
        font-weight: bold;
    }
    .strength-strong {
        color: #44aa44;
        font-weight: bold;
    }
    .password-input {
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = PasswordStrengthModel()
        model.load_model('models/random_forest_model.pkl')
        return model
    except FileNotFoundError:
        # If no saved model, train a new one
        st.warning("No pre-trained model found. Training a new model...")
        
        from data_generator import create_sample_dataset
        
        # Create and train model
        model = PasswordStrengthModel('random_forest', use_adversarial=True)
        dataset = create_sample_dataset()
        X, y, data = model.load_and_prepare_data(df=dataset)
        model.train(X, y)
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        model.save_model('models/random_forest_model.pkl')
        
        return model


def estimate_crack_time(password):
    """Estimate crack time for visualization"""
    generator = PasswordDataGenerator()
    crack_time_seconds = generator.calculate_crack_time(password)
    
    # Convert to human readable format
    if crack_time_seconds < 60:
        return f"{crack_time_seconds:.3f} seconds"
    elif crack_time_seconds < 3600:
        return f"{crack_time_seconds/60:.1f} minutes"
    elif crack_time_seconds < 86400:
        return f"{crack_time_seconds/3600:.1f} hours"
    elif crack_time_seconds < 31536000:
        return f"{crack_time_seconds/86400:.1f} days"
    elif crack_time_seconds < 31536000000:
        return f"{crack_time_seconds/31536000:.1f} years"
    else:
        return f"{crack_time_seconds/31536000:.0e} years"


def create_strength_gauge(strength, confidence):
    """Create a gauge chart for password strength"""
    strength_values = {"Weak": 1, "Medium": 2, "Strong": 3}
    value = strength_values.get(strength, 1)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Password Strength"},
        delta = {'reference': 2},
        gauge = {
            'axis': {'range': [None, 3]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "gray"},
                {'range': [2, 3], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_probability_chart(probabilities):
    """Create probability distribution chart"""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ['#ff4444', '#ff8800', '#44aa44']
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Strength Probability Distribution",
        xaxis_title="Strength Category",
        yaxis_title="Probability",
        height=300
    )
    
    return fig


def create_feature_radar_chart(features):
    """Create radar chart for password features"""
    # Select key features for radar chart
    selected_features = {
        'Length Score': min(features.get('length', 0) / 20, 1),
        'Character Diversity': features.get('unique_char_ratio', 0),
        'Entropy': min(features.get('entropy', 0) / 5, 1),
        'Complexity Score': min(features.get('complexity_score', 0) / 100, 1),
        'Case Mix': (features.get('uppercase_ratio', 0) + features.get('lowercase_ratio', 0)) / 2,
        'Symbol Usage': features.get('symbol_ratio', 0)
    }
    
    categories = list(selected_features.keys())
    values = list(selected_features.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Password Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Password Feature Analysis",
        height=400
    )
    
    return fig


def generate_adversarial_examples(password):
    """Generate adversarial examples for testing"""
    adversarial_gen = AdversarialPasswordGenerator()
    variants = adversarial_gen.generate_multiple_adversarials(password, n_variants=5)
    return variants


def main():
    # Header
    st.markdown('<h1 class="main-header">🔐 Password Strength Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced ML-powered password strength analysis with adversarial robustness
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Real-time Analysis", "Batch Analysis", "Adversarial Testing", "Model Insights"]
    )
    
    # Main content based on mode
    if analysis_mode == "Real-time Analysis":
        st.header("🔍 Real-time Password Analysis")
        
        # Password input
        password = st.text_input(
            "Enter password to analyze:",
            type="password",
            placeholder="Type your password here...",
            help="Your password is processed locally and never stored."
        )
        
        if password:
            # Predict strength
            result = model.predict_password_strength(password)
            analysis = model.analyze_password_weaknesses(password)
            
            # Create layout
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                # Strength display
                strength = result['predicted_strength']
                confidence = result['confidence']
                
                if strength == "Weak":
                    st.markdown(f'<p class="strength-weak">Strength: {strength}</p>', 
                               unsafe_allow_html=True)
                elif strength == "Medium":
                    st.markdown(f'<p class="strength-medium">Strength: {strength}</p>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="strength-strong">Strength: {strength}</p>', 
                               unsafe_allow_html=True)
                
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Estimated crack time
                crack_time = estimate_crack_time(password)
                st.metric("Estimated Crack Time", crack_time)
            
            with col2:
                # Probability chart
                fig_prob = create_probability_chart(result['probabilities'])
                st.plotly_chart(fig_prob, use_container_width=True)
            
            with col3:
                # Strength gauge
                fig_gauge = create_strength_gauge(strength, confidence)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature analysis
            st.subheader("📊 Detailed Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Feature radar chart
                fig_radar = create_feature_radar_chart(analysis['feature_analysis'])
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Weaknesses and suggestions
                if analysis['weaknesses']:
                    st.subheader("⚠️ Identified Weaknesses")
                    for weakness in analysis['weaknesses']:
                        st.write(f"• {weakness}")
                
                if analysis['suggestions']:
                    st.subheader("💡 Improvement Suggestions")
                    for suggestion in analysis['suggestions']:
                        st.write(f"• {suggestion}")
    
    elif analysis_mode == "Batch Analysis":
        st.header("📋 Batch Password Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with passwords",
            type=['csv'],
            help="CSV should have a 'Password' column"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'Password' in df.columns:
                st.success(f"Loaded {len(df)} passwords for analysis")
                
                # Analyze passwords
                if st.button("Analyze All Passwords"):
                    progress_bar = st.progress(0)
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
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        weak_count = len(results_df[results_df['Predicted_Strength'] == 'Weak'])
                        st.metric("Weak Passwords", weak_count)
                    
                    with col2:
                        medium_count = len(results_df[results_df['Predicted_Strength'] == 'Medium'])
                        st.metric("Medium Passwords", medium_count)
                    
                    with col3:
                        strong_count = len(results_df[results_df['Predicted_Strength'] == 'Strong'])
                        st.metric("Strong Passwords", strong_count)
                    
                    # Distribution chart
                    strength_dist = results_df['Predicted_Strength'].value_counts()
                    fig_dist = px.pie(
                        values=strength_dist.values,
                        names=strength_dist.index,
                        title="Password Strength Distribution"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="password_analysis_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error("CSV file must contain a 'Password' column")
    
    elif analysis_mode == "Adversarial Testing":
        st.header("🎭 Adversarial Robustness Testing")
        
        st.markdown("""
        Test how well the model handles adversarial examples - passwords that look stronger 
        than they actually are due to predictable transformations.
        """)
        
        # Input password for adversarial testing
        base_password = st.text_input(
            "Enter a base password to generate adversarial examples:",
            placeholder="e.g., password"
        )
        
        if base_password:
            # Generate adversarial examples
            adversarials = generate_adversarial_examples(base_password)
            
            st.subheader("🔄 Generated Adversarial Examples")
            
            # Analyze original and adversarial passwords
            original_result = model.predict_password_strength(base_password)
            
            results_data = []
            results_data.append({
                'Password': base_password,
                'Type': 'Original',
                'Predicted_Strength': original_result['predicted_strength'],
                'Confidence': original_result['confidence']
            })
            
            for i, adversarial in enumerate(adversarials):
                adv_result = model.predict_password_strength(adversarial)
                results_data.append({
                    'Password': adversarial,
                    'Type': f'Adversarial {i+1}',
                    'Predicted_Strength': adv_result['predicted_strength'],
                    'Confidence': adv_result['confidence']
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df)
            
            # Robustness analysis
            original_strength = original_result['predicted_strength']
            adversarial_strengths = [model.predict_password_strength(adv)['predicted_strength'] 
                                   for adv in adversarials]
            
            consistent_predictions = sum(1 for strength in adversarial_strengths 
                                       if strength == original_strength)
            robustness_score = consistent_predictions / len(adversarial_strengths)
            
            st.metric("Robustness Score", f"{robustness_score:.1%}")
            
            if robustness_score < 0.8:
                st.warning("⚠️ Model shows potential vulnerability to adversarial examples")
            else:
                st.success("✅ Model demonstrates good adversarial robustness")
    
    elif analysis_mode == "Model Insights":
        st.header("🧠 Model Insights and Performance")
        
        # Model information
        st.subheader("📊 Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model.model_type.title())
        
        with col2:
            st.metric("Adversarial Training", "Yes" if model.use_adversarial else "No")
        
        with col3:
            st.metric("Feature Count", len(model.feature_columns))
        
        # Feature importance (if available)
        if hasattr(model.model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            importances = model.model.feature_importances_
            feature_names = model.feature_columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Show top 15 features
            top_features = importance_df.head(15)
            
            fig_importance = px.bar(
                top_features, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Sample predictions
        st.subheader("🔬 Sample Predictions")
        
        sample_passwords = [
            "123456",
            "password",
            "P@ssw0rd123!",
            "MySecurePassword2024!",
            "correcthorsebatterystaple",
            "Tr0ub4dor&3",
            "qwerty123",
            "G7$kL9#mN2@pQ5"
        ]
        
        sample_results = []
        for pwd in sample_passwords:
            result = model.predict_password_strength(pwd)
            sample_results.append({
                'Password': pwd,
                'Predicted_Strength': result['predicted_strength'],
                'Confidence': f"{result['confidence']:.3f}",
                'Weak_Prob': f"{result['probabilities'].get('Weak', 0):.3f}",
                'Medium_Prob': f"{result['probabilities'].get('Medium', 0):.3f}",
                'Strong_Prob': f"{result['probabilities'].get('Strong', 0):.3f}"
            })
        
        sample_df = pd.DataFrame(sample_results)
        st.dataframe(sample_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔐 Password Strength Analyzer with Adversarial Training</p>
        <p>Built with Machine Learning for robust password security assessment</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
