#!/usr/bin/env python3
"""
Simple CTI-NLP Training Test
Test the basic functionality of our CTI-NLP system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    dirs = ['models/saved', 'logs', 'results']
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_and_explore_data():
    """Load and explore the dataset"""
    print("🔍 Loading and exploring dataset...")
    
    df = pd.read_csv('data/Cybersecurity_Dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check unique threat categories
    print(f"\nThreat Categories: {df['Threat Category'].unique()}")
    print(f"Category counts:\n{df['Threat Category'].value_counts()}")
    
    # Check severity scores
    print(f"\nSeverity Score range: {df['Severity Score'].min()} - {df['Severity Score'].max()}")
    
    return df

def preprocess_text(text):
    """Simple text preprocessing"""
    if pd.isna(text):
        return ""
    return str(text).lower()

def train_threat_classifier(df):
    """Train threat classification model"""
    print("\n🤖 Training Threat Classification Model...")
    
    # Use cleaned threat description or combine available text
    text_column = 'Cleaned Threat Description'
    if text_column not in df.columns or df[text_column].isna().all():
        # Fallback to IOCs if no threat description
        text_data = df['IOCs (Indicators of Compromise)'].apply(lambda x: str(x) if pd.notna(x) else "")
    else:
        text_data = df[text_column].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(text_data)
    y = df['Threat Category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = SGDClassifier(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nThreat Classification Results:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/saved/threat_classifier.pkl')
    joblib.dump(vectorizer, 'models/saved/threat_vectorizer.pkl')
    
    return model, vectorizer

def train_severity_predictor(df):
    """Train severity prediction model"""
    print("\n📊 Training Severity Prediction Model...")
    
    # Use word count and other features for severity prediction
    features = []
    if 'Word Count' in df.columns:
        features.append('Word Count')
    
    # Create feature matrix
    X = df[features] if features else np.random.rand(len(df), 1)  # Fallback
    y = df['Severity Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nSeverity Prediction Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    joblib.dump(model, 'models/saved/severity_predictor.pkl')
    
    return model

def test_models():
    """Test the trained models"""
    print("\n🧪 Testing Trained Models...")
    
    # Load models
    try:
        threat_model = joblib.load('models/saved/threat_classifier.pkl')
        threat_vectorizer = joblib.load('models/saved/threat_vectorizer.pkl')
        severity_model = joblib.load('models/saved/severity_predictor.pkl')
        
        # Test threat classification
        test_texts = [
            "DDoS attack detected from multiple IP addresses",
            "Malicious malware file detected in system",
            "Phishing email with suspicious links",
            "Ransomware encryption detected"
        ]
        
        print("\n🔍 Test Predictions:")
        for text in test_texts:
            # Threat prediction
            text_vec = threat_vectorizer.transform([text])
            threat_pred = threat_model.predict(text_vec)[0]
            
            # Severity prediction (using text length as feature)
            severity_features = [[len(text.split())]]  # Word count
            severity_pred = severity_model.predict(severity_features)[0]
            
            print(f"Text: '{text}'")
            print(f"  → Threat: {threat_pred}")
            print(f"  → Severity: {severity_pred:.2f}")
            print()
        
        print("✅ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False

def main():
    """Main training function"""
    print("🚀 CTI-NLP System Training Started")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    try:
        # Load data
        df = load_and_explore_data()
        
        # Train models
        threat_model, threat_vectorizer = train_threat_classifier(df)
        severity_model = train_severity_predictor(df)
        
        # Test models
        test_models()
        
        print("\n🎉 Training completed successfully!")
        print("=" * 50)
        print("📁 Saved models:")
        print("  • models/saved/threat_classifier.pkl")
        print("  • models/saved/threat_vectorizer.pkl") 
        print("  • models/saved/severity_predictor.pkl")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()