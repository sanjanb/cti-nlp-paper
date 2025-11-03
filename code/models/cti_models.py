"""
CTI-NLP Machine Learning Models
Implementation of threat classification, severity prediction, and NER models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import joblib
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatClassificationModel:
    """
    Threat Classification using various ML algorithms
    """
    
    def __init__(self):
        self.models = {
            'sgd_count': SGDClassifier(random_state=42, max_iter=1000),
            'sgd_tfidf': SGDClassifier(random_state=42, max_iter=1000),
            'logistic_count': LogisticRegression(random_state=42, max_iter=1000),
            'logistic_tfidf': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        logger.info("Training all classification models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                training_time = time.time() - start_time
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'predictions': y_pred,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Time: {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.training_results = results
        
        # Find best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name
            
            logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.3f}")
        
        return results
    
    def get_model_comparison(self):
        """Get formatted comparison of all models"""
        if not self.training_results:
            return None
        
        comparison_data = []
        for model_name, results in self.training_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.3f}",
                'F1_Score': f"{results['f1_score']:.3f}",
                'CV_Mean': f"{results['cv_mean']:.3f}",
                'CV_Std': f"{results['cv_std']:.3f}",
                'Training_Time': f"{results['training_time']:.2f}s"
            })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            # For models without predict_proba, return decision function
            if hasattr(self.best_model, 'decision_function'):
                return self.best_model.decision_function(X)
            else:
                raise ValueError("Model doesn't support probability prediction")

class SeverityPredictionModel:
    """
    Severity prediction using regression/classification approaches
    """
    
    def __init__(self):
        self.models = {
            'sgd': SGDClassifier(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.results = {}
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train severity prediction models"""
        logger.info("Training severity prediction models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                training_time = time.time() - start_time
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'training_time': training_time
                }
                
                logger.info(f"Severity {model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                
            except Exception as e:
                logger.error(f"Error training severity model {model_name}: {e}")
        
        self.results = results
        
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            self.best_model = results[best_model_name]['model']
        
        return results

class NamedEntityRecognitionModel:
    """
    Named Entity Recognition for cybersecurity entities
    """
    
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BERT-based NER model"""
        try:
            logger.info(f"Loading NER model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("ner", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    aggregation_strategy="simple")
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            # Fallback to simple pattern-based extraction
            self.pipeline = None
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        if self.pipeline:
            try:
                entities = self.pipeline(text)
                return self._process_entities(entities)
            except Exception as e:
                logger.error(f"Error in NER extraction: {e}")
                return self._fallback_entity_extraction(text)
        else:
            return self._fallback_entity_extraction(text)
    
    def _process_entities(self, entities):
        """Process entities from BERT model"""
        processed = []
        for entity in entities:
            processed.append({
                'text': entity['word'],
                'label': entity['entity_group'],
                'confidence': entity['score'],
                'start': entity.get('start', 0),
                'end': entity.get('end', 0)
            })
        return processed
    
    def _fallback_entity_extraction(self, text):
        """Fallback pattern-based entity extraction"""
        import re
        
        entities = []
        
        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)
        for ip in ips:
            entities.append({
                'text': ip,
                'label': 'IP_ADDRESS',
                'confidence': 0.9,
                'start': text.find(ip),
                'end': text.find(ip) + len(ip)
            })
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({
                'text': email,
                'label': 'EMAIL',
                'confidence': 0.9,
                'start': text.find(email),
                'end': text.find(email) + len(email)
            })
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        for url in urls:
            entities.append({
                'text': url,
                'label': 'URL',
                'confidence': 0.8,
                'start': text.find(url),
                'end': text.find(url) + len(url)
            })
        
        # CVE identifiers
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cves = re.findall(cve_pattern, text, re.IGNORECASE)
        for cve in cves:
            entities.append({
                'text': cve,
                'label': 'CVE',
                'confidence': 0.95,
                'start': text.find(cve),
                'end': text.find(cve) + len(cve)
            })
        
        return entities

class CTINLPPipeline:
    """
    Complete CTI-NLP processing pipeline
    """
    
    def __init__(self):
        self.threat_classifier = ThreatClassificationModel()
        self.severity_predictor = SeverityPredictionModel()
        self.ner_model = NamedEntityRecognitionModel()
        self.is_trained = False
    
    def train(self, classification_data, severity_data=None):
        """Train all models in the pipeline"""
        logger.info("Starting CTI-NLP pipeline training...")
        
        # Train threat classification
        classification_results = self.threat_classifier.train_all_models(
            classification_data['X_train'],
            classification_data['y_train'],
            classification_data['X_test'],
            classification_data['y_test']
        )
        
        # Train severity prediction if data provided
        if severity_data:
            severity_results = self.severity_predictor.train(
                severity_data['X_train'],
                severity_data['y_train'],
                severity_data['X_test'],
                severity_data['y_test']
            )
        
        self.is_trained = True
        logger.info("CTI-NLP pipeline training completed")
        
        return {
            'classification': classification_results,
            'severity': severity_data and self.severity_predictor.results
        }
    
    def analyze_threat(self, text):
        """Complete threat analysis pipeline"""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        results = {
            'input_text': text,
            'timestamp': time.time()
        }
        
        # Entity extraction
        entities = self.ner_model.extract_entities(text)
        results['entities'] = entities
        
        # Note: For complete classification, we'd need to vectorize the input text
        # This would require the same preprocessing pipeline used during training
        
        return results
    
    def save_models(self, base_path="models/"):
        """Save trained models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        if self.threat_classifier.best_model:
            joblib.dump(self.threat_classifier.best_model, f"{base_path}/threat_classifier.pkl")
        
        if self.severity_predictor.best_model:
            joblib.dump(self.severity_predictor.best_model, f"{base_path}/severity_predictor.pkl")
        
        logger.info(f"Models saved to {base_path}")
    
    def load_models(self, base_path="models/"):
        """Load pre-trained models"""
        try:
            self.threat_classifier.best_model = joblib.load(f"{base_path}/threat_classifier.pkl")
            self.severity_predictor.best_model = joblib.load(f"{base_path}/severity_predictor.pkl")
            self.is_trained = True
            logger.info(f"Models loaded from {base_path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

if __name__ == "__main__":
    # Example usage
    logger.info("CTI-NLP Models Module Initialized")