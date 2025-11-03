#!/usr/bin/env python3
"""
Simple CTI-NLP API Test
Test the FastAPI backend functionality
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from pathlib import Path
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="CTI-NLP API", description="Cyber Threat Intelligence NLP Analysis", version="1.0.0")

# Request/Response models
class ThreatAnalysisRequest(BaseModel):
    text: str

class ThreatAnalysisResponse(BaseModel):
    threat_category: str
    severity_score: float
    confidence: float
    iocs_detected: list
    analysis_summary: str

# Global variables for models
threat_classifier = None
threat_vectorizer = None
severity_predictor = None

def load_models():
    """Load trained models"""
    global threat_classifier, threat_vectorizer, severity_predictor
    
    models_dir = Path("models/saved")
    
    try:
        if (models_dir / "threat_classifier.pkl").exists():
            threat_classifier = joblib.load(models_dir / "threat_classifier.pkl")
            threat_vectorizer = joblib.load(models_dir / "threat_vectorizer.pkl")
            print("✅ Threat classification models loaded")
        
        if (models_dir / "severity_predictor.pkl").exists():
            severity_predictor = joblib.load(models_dir / "severity_predictor.pkl")
            print("✅ Severity prediction model loaded")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")

def extract_simple_iocs(text):
    """Simple IOC extraction (IP addresses, domains)"""
    import re
    
    # Simple patterns
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    domain_pattern = r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.(?:[a-zA-Z]{2,})\b'
    
    iocs = []
    
    # Find IPs
    ips = re.findall(ip_pattern, text)
    for ip in ips:
        iocs.append({"type": "ip_address", "value": ip})
    
    # Find domains
    domains = re.findall(domain_pattern, text)
    for domain in domains:
        if not any(domain.endswith(common) for common in ['.com', '.org', '.net', '.edu']):
            continue
        iocs.append({"type": "domain", "value": domain})
    
    return iocs

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting CTI-NLP API...")
    load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CTI-NLP API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "threat_classifier": threat_classifier is not None,
            "severity_predictor": severity_predictor is not None
        }
    }

@app.post("/analyze", response_model=ThreatAnalysisResponse)
async def analyze_threat(request: ThreatAnalysisRequest):
    """Analyze a threat text"""
    try:
        text = request.text
        
        # Default values
        threat_category = "Unknown"
        severity_score = 3.0
        confidence = 0.5
        
        # Threat classification
        if threat_classifier and threat_vectorizer:
            try:
                text_vec = threat_vectorizer.transform([text])
                threat_category = threat_classifier.predict(text_vec)[0]
                
                # Get prediction probabilities for confidence
                if hasattr(threat_classifier, 'predict_proba'):
                    proba = threat_classifier.predict_proba(text_vec)[0]
                    confidence = float(max(proba))
            except Exception as e:
                print(f"Threat classification error: {e}")
        
        # Severity prediction
        if severity_predictor:
            try:
                # Use word count as feature
                word_count = len(text.split())
                severity_features = [[word_count]]
                severity_score = float(severity_predictor.predict(severity_features)[0])
                # Clamp to valid range
                severity_score = max(1.0, min(5.0, severity_score))
            except Exception as e:
                print(f"Severity prediction error: {e}")
        
        # Extract IOCs
        iocs_detected = extract_simple_iocs(text)
        
        # Create analysis summary
        analysis_summary = f"Detected {threat_category} threat with severity {severity_score:.1f}. "
        if iocs_detected:
            analysis_summary += f"Found {len(iocs_detected)} IOCs. "
        analysis_summary += f"Confidence: {confidence:.2f}"
        
        return ThreatAnalysisResponse(
            threat_category=threat_category,
            severity_score=severity_score,
            confidence=confidence,
            iocs_detected=iocs_detected,
            analysis_summary=analysis_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_threats_batch(texts: list[str]):
    """Analyze multiple threats"""
    results = []
    for text in texts:
        try:
            request = ThreatAnalysisRequest(text=text)
            result = await analyze_threat(request)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e), "text": text})
    
    return {"results": results, "total": len(texts)}

if __name__ == "__main__":
    print("🚀 Starting CTI-NLP API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)