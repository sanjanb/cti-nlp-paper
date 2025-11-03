"""
CTI-NLP Data Preprocessing Module
Handles data cleaning, feature extraction, and preparation for ML models
"""

import pandas as pd
import numpy as np
import re
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTIDataPreprocessor:
    """
    Comprehensive data preprocessor for CTI-NLP system
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.vectorizers = {
            'count': CountVectorizer(max_features=5000, stop_words='english'),
            'tfidf': TfidfVectorizer(max_features=5000, stop_words='english')
        }
        
    def load_data(self, filepath):
        """Load and inspect the cybersecurity dataset"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_iocs(self, ioc_string):
        """Extract and parse IOCs from string representation"""
        try:
            if pd.isna(ioc_string):
                return []
            
            # Handle string representation of list
            if isinstance(ioc_string, str):
                # Remove quotes and brackets, split by comma
                iocs = re.findall(r"'([^']*)'", ioc_string)
                if not iocs:
                    # Try alternative parsing
                    iocs = [x.strip() for x in ioc_string.strip("[]'\"").split(',')]
                return [ioc.strip() for ioc in iocs if ioc.strip()]
            
            return ioc_string if isinstance(ioc_string, list) else []
        except:
            return []
    
    def preprocess_dataset(self, df):
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text columns
        text_columns = ['Cleaned Threat Description', 'Threat Actor', 'Attack Vector']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(self.clean_text)
        
        # Process IOCs
        if 'IOCs (Indicators of Compromise)' in processed_df.columns:
            processed_df['IOCs_parsed'] = processed_df['IOCs (Indicators of Compromise)'].apply(self.extract_iocs)
            processed_df['IOC_count'] = processed_df['IOCs_parsed'].apply(len)
        
        # Handle categorical variables
        categorical_cols = ['Threat Category', 'Threat Actor', 'Attack Vector', 'Geographical Location']
        for col in categorical_cols:
            if col in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle missing values
        processed_df = processed_df.fillna({
            'Severity Score': processed_df['Severity Score'].median() if 'Severity Score' in processed_df.columns else 0,
            'Sentiment in Forums': processed_df['Sentiment in Forums'].median() if 'Sentiment in Forums' in processed_df.columns else 0.5,
            'Risk Level Prediction': processed_df['Risk Level Prediction'].median() if 'Risk Level Prediction' in processed_df.columns else 3
        })
        
        logger.info("Data preprocessing completed")
        return processed_df
    
    def create_features(self, df, text_column='Cleaned Threat Description', vectorizer_type='count'):
        """Create feature vectors for ML models"""
        
        if text_column not in df.columns:
            logger.error(f"Column {text_column} not found in dataset")
            return None, None
        
        # Vectorize text
        vectorizer = self.vectorizers[vectorizer_type]
        X_text = vectorizer.fit_transform(df[text_column].astype(str))
        
        # Additional numerical features
        numerical_features = []
        feature_names = []
        
        if 'Severity Score' in df.columns:
            numerical_features.append(df['Severity Score'].values.reshape(-1, 1))
            feature_names.append('severity_score')
        
        if 'Sentiment in Forums' in df.columns:
            numerical_features.append(df['Sentiment in Forums'].values.reshape(-1, 1))
            feature_names.append('sentiment_score')
        
        if 'IOC_count' in df.columns:
            numerical_features.append(df['IOC_count'].values.reshape(-1, 1))
            feature_names.append('ioc_count')
        
        if 'Word Count' in df.columns:
            numerical_features.append(df['Word Count'].values.reshape(-1, 1))
            feature_names.append('word_count')
        
        # Combine features if numerical features exist
        if numerical_features:
            from scipy.sparse import hstack
            numerical_matrix = np.hstack(numerical_features)
            X_combined = hstack([X_text, numerical_matrix])
        else:
            X_combined = X_text
        
        return X_combined, vectorizer.get_feature_names_out()
    
    def prepare_classification_data(self, df, target_column='Threat Category', test_size=0.2, random_state=42):
        """Prepare data for classification tasks"""
        
        # Create features
        X, feature_names = self.create_features(df)
        
        if X is None:
            return None
        
        # Target variable
        y = df[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Classes: {np.unique(y)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'classes': np.unique(y)
        }
    
    def get_class_distribution(self, df, column='Threat Category'):
        """Analyze class distribution"""
        if column in df.columns:
            distribution = df[column].value_counts()
            logger.info(f"Class distribution for {column}:")
            for class_name, count in distribution.items():
                logger.info(f"  {class_name}: {count} ({count/len(df)*100:.1f}%)")
            return distribution
        return None

# Data analysis utilities
def analyze_dataset(filepath):
    """Comprehensive dataset analysis"""
    preprocessor = CTIDataPreprocessor()
    df = preprocessor.load_data(filepath)
    
    if df is None:
        return None
    
    print("=" * 50)
    print("CYBERSECURITY DATASET ANALYSIS")
    print("=" * 50)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Missing values
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    # Class distributions
    categorical_cols = ['Threat Category', 'Predicted Threat Category', 'Attack Vector', 'Threat Actor']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col} Distribution:")
            print(df[col].value_counts().head(10))
    
    # Numerical statistics
    numerical_cols = ['Severity Score', 'Risk Level Prediction', 'Sentiment in Forums', 'Word Count']
    for col in numerical_cols:
        if col in df.columns:
            print(f"\n{col} Statistics:")
            print(df[col].describe())
    
    return df

if __name__ == "__main__":
    # Analyze the dataset
    dataset_path = "../data/Cybersecurity_Dataset.csv"
    df = analyze_dataset(dataset_path)