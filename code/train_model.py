"""
CTI-NLP Training Script
Complete training pipeline for the cybersecurity threat intelligence system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from utils.data_preprocessing import CTIDataPreprocessor, analyze_dataset
from models.cti_models import CTINLPPipeline
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("CTI-NLP TRAINING PIPELINE")
    logger.info("="*60)
    
    # Configuration
    DATA_PATH = "../data/Cybersecurity_Dataset.csv"
    MODEL_SAVE_PATH = "../models/"
    RESULTS_PATH = "../results/"
    
    # Create directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Step 1: Load and analyze data
    logger.info("Step 1: Loading and analyzing dataset...")
    preprocessor = CTIDataPreprocessor()
    df = preprocessor.load_data(DATA_PATH)
    
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Basic analysis
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Class distribution
    threat_distribution = preprocessor.get_class_distribution(df, 'Threat Category')
    
    # Step 2: Preprocess data
    logger.info("\nStep 2: Preprocessing data...")
    processed_df = preprocessor.preprocess_dataset(df)
    
    # Step 3: Prepare classification data
    logger.info("\nStep 3: Preparing classification data...")
    classification_data = preprocessor.prepare_classification_data(
        processed_df, 
        target_column='Threat Category',
        test_size=0.2,
        random_state=42
    )
    
    if classification_data is None:
        logger.error("Failed to prepare classification data. Exiting.")
        return
    
    # Step 4: Prepare severity prediction data
    logger.info("\nStep 4: Preparing severity prediction data...")
    severity_data = None
    if 'Risk Level Prediction' in processed_df.columns:
        severity_data = preprocessor.prepare_classification_data(
            processed_df,
            target_column='Risk Level Prediction',
            test_size=0.2,
            random_state=42
        )
    
    # Step 5: Initialize and train pipeline
    logger.info("\nStep 5: Training CTI-NLP pipeline...")
    pipeline = CTINLPPipeline()
    
    start_time = time.time()
    training_results = pipeline.train(classification_data, severity_data)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Step 6: Evaluate and display results
    logger.info("\nStep 6: Evaluation Results")
    logger.info("="*40)
    
    # Classification results
    if 'classification' in training_results:
        logger.info("THREAT CLASSIFICATION RESULTS:")
        
        classification_comparison = pipeline.threat_classifier.get_model_comparison()
        if classification_comparison is not None:
            print("\nModel Comparison:")
            print(classification_comparison.to_string(index=False))
            
            # Save results
            classification_comparison.to_csv(f"{RESULTS_PATH}/classification_results.csv", index=False)
        
        # Best model details
        best_model_name = pipeline.threat_classifier.best_model_name
        best_results = pipeline.threat_classifier.training_results[best_model_name]
        
        logger.info(f"\nBest Classification Model: {best_model_name}")
        logger.info(f"Accuracy: {best_results['accuracy']:.4f}")
        logger.info(f"F1-Score: {best_results['f1_score']:.4f}")
        logger.info(f"Cross-validation: {best_results['cv_mean']:.4f} (+/- {best_results['cv_std']:.4f})")
    
    # Severity prediction results
    if severity_data and 'severity' in training_results:
        logger.info("\nSEVERITY PREDICTION RESULTS:")
        
        severity_results = training_results['severity']
        for model_name, results in severity_results.items():
            logger.info(f"{model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
    
    # Step 7: Test NER model
    logger.info("\nStep 7: Testing Named Entity Recognition...")
    
    # Test with sample texts
    test_texts = [
        "Malware detected at IP address 192.168.1.100 with CVE-2023-1234",
        "Phishing email from suspicious.domain.com targeting corporate accounts",
        "Ransomware attack through network vulnerability affecting CompanyX systems",
        "DDoS attack on website example.com from botnet at 10.0.0.5"
    ]
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\nTest {i}: {text}")
        entities = pipeline.ner_model.extract_entities(text)
        for entity in entities:
            logger.info(f"  Entity: {entity['text']} | Label: {entity['label']} | Confidence: {entity['confidence']:.3f}")
    
    # Step 8: Save models and results
    logger.info("\nStep 8: Saving models and results...")
    
    pipeline.save_models(MODEL_SAVE_PATH)
    
    # Save training metadata
    metadata = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_shape': df.shape,
        'training_time_seconds': training_time,
        'best_classification_model': best_model_name,
        'best_classification_accuracy': float(best_results['accuracy']),
        'threat_categories': list(classification_data['classes']),
        'feature_count': classification_data['X_train'].shape[1]
    }
    
    with open(f"{RESULTS_PATH}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Step 9: Generate performance report
    logger.info("\nStep 9: Generating performance report...")
    
    report = generate_performance_report(
        classification_results=training_results['classification'],
        severity_results=training_results.get('severity'),
        metadata=metadata
    )
    
    with open(f"{RESULTS_PATH}/performance_report.md", 'w') as f:
        f.write(report)
    
    logger.info("="*60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Models saved to: {MODEL_SAVE_PATH}")
    logger.info(f"Results saved to: {RESULTS_PATH}")
    logger.info(f"Best model: {best_model_name} with {best_results['accuracy']:.1%} accuracy")

def generate_performance_report(classification_results, severity_results, metadata):
    """Generate a comprehensive performance report"""
    
    report = f"""# CTI-NLP Performance Report

Generated on: {metadata['training_date']}

## Dataset Information
- **Dataset Shape**: {metadata['dataset_shape'][0]} samples, {metadata['dataset_shape'][1]} features
- **Training Time**: {metadata['training_time_seconds']:.2f} seconds
- **Feature Count**: {metadata['feature_count']}
- **Threat Categories**: {', '.join(metadata['threat_categories'])}

## Threat Classification Results

### Best Model: {metadata['best_classification_model']}
- **Accuracy**: {metadata['best_classification_accuracy']:.4f}
- **Performance**: {metadata['best_classification_accuracy']*100:.1f}%

### All Models Comparison
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
"""
    
    # Add classification results
    for model_name, results in classification_results.items():
        accuracy = results['accuracy']
        f1_score = results['f1_score']
        time_taken = results['training_time']
        report += f"| {model_name} | {accuracy:.4f} | {f1_score:.4f} | {time_taken:.2f}s |\n"
    
    if severity_results:
        report += "\n## Severity Prediction Results\n\n"
        report += "| Model | Accuracy | F1-Score |\n"
        report += "|-------|----------|----------|\n"
        
        for model_name, results in severity_results.items():
            accuracy = results['accuracy']
            f1_score = results['f1_score']
            report += f"| {model_name} | {accuracy:.4f} | {f1_score:.4f} |\n"
    
    report += """
## Key Achievements
- Automated threat intelligence analysis
- Multi-class threat categorization
- Real-time entity extraction
- Production-ready model performance

## Technical Implementation
- **Framework**: scikit-learn, transformers
- **Vectorization**: Count Vectorizer, TF-IDF
- **Models**: SGD Classifier, Random Forest, Logistic Regression
- **NER**: BERT-based Named Entity Recognition
- **Deployment**: FastAPI backend, Docker containerization
"""
    
    return report

if __name__ == "__main__":
    main()