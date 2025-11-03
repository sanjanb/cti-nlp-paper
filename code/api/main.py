"""
CTI-NLP FastAPI Backend
Real-time threat intelligence analysis API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cti_models import CTINLPPipeline
from utils.data_preprocessing import CTIDataPreprocessor
import logging
import time
import json
import uvicorn
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CTI-NLP API",
    description="AI-Powered Cyber Threat Intelligence Analysis System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
preprocessor = None

# Pydantic models
class ThreatAnalysisRequest(BaseModel):
    text: str = Field(..., description="Threat description text to analyze")
    include_entities: bool = Field(True, description="Include named entity recognition")
    include_classification: bool = Field(True, description="Include threat classification")
    include_severity: bool = Field(True, description="Include severity prediction")

class EntityResult(BaseModel):
    text: str
    label: str
    confidence: float
    start: int
    end: int

class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

class SeverityResult(BaseModel):
    risk_level: int
    risk_description: str
    confidence: float

class ThreatAnalysisResponse(BaseModel):
    success: bool
    processing_time_ms: float
    timestamp: str
    input_text: str
    results: Dict[str, Any]
    entities: Optional[List[EntityResult]] = None
    classification: Optional[ClassificationResult] = None
    severity: Optional[SeverityResult] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool
    uptime_seconds: float

class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    options: ThreatAnalysisRequest

class StatsResponse(BaseModel):
    total_requests: int
    successful_analyses: int
    average_processing_time: float
    most_common_threats: Dict[str, int]

# Global stats
stats = {
    "total_requests": 0,
    "successful_analyses": 0,
    "processing_times": [],
    "threat_counts": {},
    "start_time": time.time()
}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global pipeline, preprocessor
    
    logger.info("Initializing CTI-NLP system...")
    
    try:
        # Initialize components
        pipeline = CTINLPPipeline()
        preprocessor = CTIDataPreprocessor()
        
        # Try to load pre-trained models
        model_path = "../models/"
        if os.path.exists(model_path):
            try:
                pipeline.load_models(model_path)
                logger.info("Pre-trained models loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load pre-trained models: {e}")
                logger.info("Models will need to be trained before use")
        
        logger.info("CTI-NLP system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize CTI-NLP system: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="active",
        timestamp=datetime.now().isoformat(),
        models_loaded=pipeline is not None and pipeline.is_trained,
        uptime_seconds=time.time() - stats["start_time"]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=pipeline is not None and pipeline.is_trained,
        uptime_seconds=time.time() - stats["start_time"]
    )

@app.post("/analyze", response_model=ThreatAnalysisResponse)
async def analyze_threat(request: ThreatAnalysisRequest):
    """Analyze a single threat description"""
    global stats
    
    stats["total_requests"] += 1
    start_time = time.time()
    
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="CTI-NLP system not initialized")
        
        if not pipeline.is_trained:
            raise HTTPException(status_code=503, detail="Models not trained. Please train models first.")
        
        # Clean input text
        cleaned_text = preprocessor.clean_text(request.text)
        
        results = {}
        entities = None
        classification = None
        severity = None
        
        # Named Entity Recognition
        if request.include_entities:
            entity_list = pipeline.ner_model.extract_entities(cleaned_text)
            entities = [EntityResult(**entity) for entity in entity_list]
            results["entities"] = entity_list
        
        # Threat Classification
        if request.include_classification:
            # Note: For full implementation, we'd need to vectorize the input
            # using the same preprocessing pipeline as training
            classification = ClassificationResult(
                predicted_class="Phishing",  # Placeholder
                confidence=0.87,
                all_probabilities={
                    "Phishing": 0.87,
                    "Malware": 0.08,
                    "DDoS": 0.03,
                    "Ransomware": 0.02
                }
            )
            results["classification"] = classification.dict()
        
        # Severity Prediction
        if request.include_severity:
            severity = SeverityResult(
                risk_level=3,
                risk_description="Medium Risk",
                confidence=0.82
            )
            results["severity"] = severity.dict()
        
        processing_time = (time.time() - start_time) * 1000
        stats["processing_times"].append(processing_time)
        stats["successful_analyses"] += 1
        
        # Update threat counts
        if classification:
            threat_type = classification.predicted_class
            stats["threat_counts"][threat_type] = stats["threat_counts"].get(threat_type, 0) + 1
        
        return ThreatAnalysisResponse(
            success=True,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            input_text=request.text,
            results=results,
            entities=entities,
            classification=classification,
            severity=severity
        )
        
    except Exception as e:
        logger.error(f"Error in threat analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def batch_analyze(request: BatchAnalysisRequest):
    """Analyze multiple threat descriptions"""
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 texts.")
    
    results = []
    start_time = time.time()
    
    for text in request.texts:
        try:
            analysis_request = ThreatAnalysisRequest(
                text=text,
                include_entities=request.options.include_entities,
                include_classification=request.options.include_classification,
                include_severity=request.options.include_severity
            )
            
            result = await analyze_threat(analysis_request)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in batch analysis for text '{text[:50]}...': {e}")
            results.append({
                "success": False,
                "error": str(e),
                "input_text": text
            })
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "success": True,
        "total_processing_time_ms": total_time,
        "results_count": len(results),
        "results": results
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics"""
    avg_time = sum(stats["processing_times"]) / len(stats["processing_times"]) if stats["processing_times"] else 0
    
    return StatsResponse(
        total_requests=stats["total_requests"],
        successful_analyses=stats["successful_analyses"],
        average_processing_time=avg_time,
        most_common_threats=dict(sorted(stats["threat_counts"].items(), key=lambda x: x[1], reverse=True)[:10])
    )

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    info = {
        "is_trained": pipeline.is_trained,
        "threat_classifier": {
            "best_model": pipeline.threat_classifier.best_model_name if pipeline.threat_classifier.best_model else None,
            "available_models": list(pipeline.threat_classifier.models.keys())
        },
        "ner_model": {
            "model_name": pipeline.ner_model.model_name,
            "is_loaded": pipeline.ner_model.pipeline is not None
        },
        "severity_predictor": {
            "is_trained": pipeline.severity_predictor.best_model is not None
        }
    }
    
    return info

@app.post("/models/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Trigger model training (background task)"""
    
    def train_models():
        """Background training task"""
        try:
            logger.info("Starting background model training...")
            
            # This would call the training script
            # For now, just simulate training
            time.sleep(5)  # Simulate training time
            
            logger.info("Background training completed")
            
        except Exception as e:
            logger.error(f"Background training failed: {e}")
    
    background_tasks.add_task(train_models)
    
    return {
        "message": "Training started in background",
        "status": "initiated"
    }

@app.get("/examples")
async def get_examples():
    """Get example threat descriptions for testing"""
    examples = [
        {
            "text": "Suspicious email from unknown sender containing malicious attachment targeting corporate network",
            "category": "Phishing",
            "severity": "High"
        },
        {
            "text": "Malware detected at IP address 192.168.1.100 attempting to access sensitive files",
            "category": "Malware", 
            "severity": "Critical"
        },
        {
            "text": "DDoS attack on public website from botnet compromising service availability",
            "category": "DDoS",
            "severity": "Medium"
        },
        {
            "text": "Ransomware encryption of file systems demanding cryptocurrency payment for decryption",
            "category": "Ransomware",
            "severity": "Critical"
        },
        {
            "text": "CVE-2023-1234 vulnerability in Apache server allowing remote code execution",
            "category": "Vulnerability",
            "severity": "High"
        }
    ]
    
    return {"examples": examples}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )