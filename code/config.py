# CTI-NLP System Configuration

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "csv_file": DATA_DIR / "Cybersecurity_Dataset.csv",
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# Model configuration
MODEL_CONFIG = {
    "threat_classification": {
        "algorithms": [
            "sgd", "random_forest", "svm", "logistic_regression",
            "naive_bayes", "gradient_boosting", "ada_boost",
            "extra_trees", "knn"
        ],
        "vectorizer": {
            "max_features": 10000,
            "ngram_range": (1, 3),
            "stop_words": "english"
        }
    },
    "severity_prediction": {
        "algorithm": "random_forest",
        "n_estimators": 100,
        "random_state": 42
    },
    "ner": {
        "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "max_length": 512,
        "batch_size": 16
    }
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "cors_origins": ["*"],
    "max_request_size": 1024 * 1024,  # 1MB
    "rate_limit": {
        "requests": 100,
        "window": 60  # seconds
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "cti_nlp.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# IOC patterns for extraction
IOC_PATTERNS = {
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "domain": r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b",
    "url": r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "hash_md5": r"\b[a-fA-F0-9]{32}\b",
    "hash_sha1": r"\b[a-fA-F0-9]{40}\b",
    "hash_sha256": r"\b[a-fA-F0-9]{64}\b",
    "cve": r"CVE-\d{4}-\d{4,7}",
    "file_path": r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*",
    "registry_key": r"HKEY_[A-Z_]+\\(?:[^\\]+\\)*[^\\]*"
}

# Threat categories mapping
THREAT_CATEGORIES = {
    "DDoS": "Distributed Denial of Service",
    "Malware": "Malicious Software", 
    "Phishing": "Social Engineering Attack",
    "Ransomware": "Encryption-based Extortion",
    "Data Breach": "Unauthorized Data Access",
    "APT": "Advanced Persistent Threat",
    "Insider Threat": "Internal Security Risk",
    "Zero-day": "Unknown Vulnerability Exploit"
}

# Severity levels
SEVERITY_LEVELS = {
    "Low": 1,
    "Medium": 2, 
    "High": 3,
    "Critical": 4
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.80,
    "f1_score": 0.80
}

# Environment variables with defaults
ENV_VARS = {
    "CTI_NLP_ENV": os.getenv("CTI_NLP_ENV", "development"),
    "CTI_NLP_LOG_LEVEL": os.getenv("CTI_NLP_LOG_LEVEL", "INFO"),
    "CTI_NLP_API_KEY": os.getenv("CTI_NLP_API_KEY", ""),
    "CTI_NLP_DATABASE_URL": os.getenv("CTI_NLP_DATABASE_URL", ""),
    "CTI_NLP_REDIS_URL": os.getenv("CTI_NLP_REDIS_URL", ""),
}