# CTI-NLP System Documentation

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- 2GB free disk space

### Installation

1. **Run Setup Script** (Recommended)

```bash
python setup.py
```

2. **Manual Installation**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models/saved logs data/processed results
```

### Usage

#### 1. Train Models

```bash
python train_model.py
```

This will:

- Load and preprocess the cybersecurity dataset
- Train multiple ML algorithms
- Evaluate model performance
- Save trained models and results

#### 2. Start API Server

```bash
python api/main.py
# or
uvicorn api.main:app --reload
```

Access API documentation at: http://localhost:8000/docs

#### 3. Use Dashboard

Open `frontend/dashboard.html` in your web browser for the interactive interface.

## 📁 Project Structure

```
code/
├── utils/
│   └── data_preprocessing.py    # Data preprocessing pipeline
├── models/
│   ├── cti_models.py           # ML model implementations
│   └── saved/                  # Trained model files
├── api/
│   └── main.py                 # FastAPI backend
├── frontend/
│   └── dashboard.html          # Web dashboard
├── logs/                       # Application logs
├── results/                    # Training results
├── config.py                   # Configuration settings
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── setup.py                  # Setup automation
```

## 🔧 Configuration

Edit `config.py` to customize:

- Model parameters
- API settings
- File paths
- Performance thresholds

## 📊 Dataset

The system uses `Cybersecurity_Dataset.csv` with:

- **1,102 samples** across threat categories
- **Columns:** Threat Category, IOCs, Severity, Description
- **Categories:** DDoS, Malware, Phishing, Ransomware

## 🤖 Models

### 1. Threat Classification

- **Algorithms:** SGD, Random Forest, SVM, Logistic Regression, etc.
- **Features:** TF-IDF vectorization with n-grams
- **Target:** Predict threat category

### 2. Severity Prediction

- **Algorithm:** Random Forest Regressor
- **Features:** Text embeddings + IOC patterns
- **Target:** Predict severity score (1-4)

### 3. Named Entity Recognition

- **Model:** BERT-large-cased fine-tuned on CoNLL-03
- **Entities:** IP addresses, domains, file paths, CVEs
- **Output:** IOC extraction and classification

## 🌐 API Endpoints

### Single Analysis

```bash
POST /analyze
{
  "text": "Malicious IP 192.168.1.100 detected"
}
```

### Batch Analysis

```bash
POST /analyze/batch
{
  "texts": ["threat1", "threat2", "threat3"]
}
```

### Health Check

```bash
GET /health
```

### Statistics

```bash
GET /stats
```

## 📈 Performance Metrics

Expected performance on test set:

- **Accuracy:** >85%
- **Precision:** >80%
- **Recall:** >80%
- **F1-Score:** >80%

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Memory Issues**

   - Reduce batch size in config
   - Use smaller model variants
   - Increase system RAM

3. **Model Download Fails**

   ```bash
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-cased')"
   ```

4. **API Won't Start**
   - Check port 8000 availability
   - Verify virtual environment activation
   - Check logs in `logs/cti_nlp.log`

### Debug Mode

```bash
export CTI_NLP_LOG_LEVEL=DEBUG
python train_model.py
```

## 🔒 Security Considerations

- API rate limiting enabled
- Input validation and sanitization
- No sensitive data in logs
- Model artifacts secured

## 📝 Development

### Adding New Models

1. Extend `CTINLPPipeline` in `models/cti_models.py`
2. Update training pipeline in `train_model.py`
3. Add API endpoints in `api/main.py`

### Custom Preprocessing

1. Modify `CTIDataPreprocessor` in `utils/data_preprocessing.py`
2. Update IOC patterns in `config.py`
3. Retrain models

### Testing

```bash
pytest tests/ -v
```

## 📊 Monitoring

- **Logs:** `logs/cti_nlp.log`
- **Metrics:** Training results in `results/`
- **Models:** Saved in `models/saved/`
- **API Health:** http://localhost:8000/health

## 🚀 Production Deployment

### Docker (Recommended)

```bash
docker build -t cti-nlp .
docker run -p 8000:8000 cti-nlp
```

### Manual Deployment

```bash
# Install production server
pip install gunicorn

# Start with gunicorn
gunicorn api.main:app --workers 4 --bind 0.0.0.0:8000
```

## 📚 References

1. **BERT for NER:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
2. **SGD Classification:** Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent"
3. **CTI Analysis:** Samtani et al., "Exploring Emerging Hacker Assets and Key Hackers"

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## 📄 License

Academic research use only. See LICENSE file for details.

## 📞 Support

For issues and questions:

- Check troubleshooting section
- Review logs in `logs/`
- Open GitHub issue with error details
