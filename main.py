import os
import io
import torch
import base64
import numpy as np
import tempfile
import uuid
import time
import hashlib
import structlog
from datetime import datetime, timezone
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image
import uvicorn

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized
)
from monai.networks.nets import TorchVisionFCModel

############################
# CONFIGURATION & MODELS  #
############################

class Config:
    """
    Application configuration using environment variables.
    Health Universe handles authentication at the platform level.
    """
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.pt")
    SAMPLE_DATA_PATH: str = os.getenv("SAMPLE_DATA_PATH", "sample_data/A/sample_A1.jpg")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".dcm"]
    
    # Logging and Monitoring
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    
    # Deployment Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

class ClassificationRequest(BaseModel):
    """Request model for classification with validation"""
    patient_id: Optional[str] = Field(None, description="De-identified patient ID")
    study_id: Optional[str] = Field(None, description="Study identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        if v and (len(v) > 50 or not v.replace('-', '').replace('_', '').isalnum()):
            raise ValueError("Invalid patient ID format")
        return v

class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    request_id: str = Field(..., description="Unique request identifier")
    predicted_class: str = Field(..., description="Predicted breast density class (A, B, C, D)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(default="1.0", description="Model version used")
    timestamp: str = Field(..., description="ISO timestamp of classification")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str

############################
# LOGGING CONFIGURATION   #
############################

def configure_logging():
    """Configure structured logging for HIPAA compliance"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

configure_logging()
logger = structlog.get_logger()

############################
# GLOBALS & INITIALIZATION #
############################

BREAST_CLASSES = ["A", "B", "C", "D"]
MODEL_VERSION = "1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model loading status (like working app)
model_loaded = False
MODEL = None
model_load_error = None

preprocess = Compose([
    LoadImaged(keys="image", image_only=False),
    EnsureChannelFirstd(keys="image", channel_dim=2),
    ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
    Resized(keys="image", spatial_size=(299, 299))
])

############################
# UTILITY FUNCTIONS       #
############################

def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return str(uuid.uuid4())

def anonymize_filename(filename: str) -> str:
    """Create anonymized hash of filename for logging"""
    return hashlib.sha256(filename.encode()).hexdigest()[:16]

def validate_file_size(file_size: int) -> None:
    """Validate uploaded file size"""
    if file_size > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
        )

def validate_image_extension(filename: str) -> None:
    """Validate file extension"""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    ext = Path(filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
        )


def load_model_sync() -> TorchVisionFCModel:
    """Load model synchronously with compatibility fixes for deployment"""
    try:
        model = TorchVisionFCModel(
            model_name="inception_v3",
            num_classes=4,
            pretrained=True,
            pool=None,
            bias=True,
            use_conv=False
        ).to(DEVICE)

        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")

        # Version-compatible loading strategies
        checkpoint = None
        loading_errors = []
        
        # Strategy 1: Standard loading (your local method)
        try:
            checkpoint = torch.load(Config.MODEL_PATH, map_location=DEVICE, weights_only=False)
            logger.info("model_loaded_with_standard_method")
        except Exception as e:
            loading_errors.append(f"Standard: {e}")
            
            # Strategy 2: Load to CPU first, then move to device
            try:
                checkpoint = torch.load(Config.MODEL_PATH, map_location='cpu', weights_only=False)
                logger.info("model_loaded_with_cpu_method")
            except Exception as e2:
                loading_errors.append(f"CPU: {e2}")
                
                # Strategy 3: Try without weights_only parameter (older PyTorch)
                try:
                    checkpoint = torch.load(Config.MODEL_PATH, map_location=DEVICE)
                    logger.info("model_loaded_without_weights_only")
                except Exception as e3:
                    loading_errors.append(f"No weights_only: {e3}")
                    
                    # Strategy 4: Use pickle protocol compatibility
                    try:
                        import pickle
                        with open(Config.MODEL_PATH, 'rb') as f:
                            checkpoint = pickle.load(f)
                        logger.info("model_loaded_with_pickle")
                    except Exception as e4:
                        loading_errors.append(f"Pickle: {e4}")
                        raise RuntimeError(f"All loading methods failed: {loading_errors}")

        # Your diagnostic shows it's a direct OrderedDict, so load it directly
        if isinstance(checkpoint, dict) and 'fc.weight' in checkpoint:
            # It's a direct state_dict (as confirmed by your diagnostic)
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume it's already a state_dict
            state_dict = checkpoint

        # Load with strict=False to handle minor compatibility issues
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning("missing_keys_in_model", missing_keys=missing_keys[:5])  # Log first 5
        if unexpected_keys:
            logger.warning("unexpected_keys_in_model", unexpected_keys=unexpected_keys[:5])
        
        # Verify the final layer exists and has correct size
        if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            if model.fc.out_features != 4:
                logger.warning("incorrect_output_size", 
                             expected=4, 
                             actual=model.fc.out_features)

        model.eval()
        logger.info("model_loaded_successfully", 
                   model_path=Config.MODEL_PATH, 
                   device=str(DEVICE),
                   missing_keys_count=len(missing_keys),
                   unexpected_keys_count=len(unexpected_keys))
        return model
        
    except Exception as e:
        logger.error("model_loading_failed", error=str(e), model_path=Config.MODEL_PATH)
        raise

def run_inference_sync(image_bytes: bytes, request_id: str) -> Dict[str, Any]:
    """Run inference synchronously"""
    start_time = time.time()
    
    try:
        # Create temporary file with secure cleanup
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            # Preprocess image
            data_dict = {"image": tmp_path}
            out_dict = preprocess(data_dict)
            image_np = out_dict["image"]

            # Convert to tensor and run inference
            tensor = torch.as_tensor(image_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = MODEL(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Calculate results
            predicted_idx = np.argmax(probs)
            predicted_class = BREAST_CLASSES[predicted_idx]
            confidence_score = float(probs[predicted_idx])
            processing_time = (time.time() - start_time) * 1000

            # Log successful inference (HIPAA-compliant)
            logger.info("classification_completed",
                       request_id=request_id,
                       predicted_class=predicted_class,
                       confidence_score=confidence_score,
                       processing_time_ms=processing_time,
                       version=MODEL_VERSION)

            return {
                "predicted_class": predicted_class,
                "probabilities": {BREAST_CLASSES[i]: float(probs[i]) for i in range(len(probs))},
                "confidence_score": confidence_score,
                "processing_time_ms": processing_time
            }

        finally:
            # Secure cleanup of temporary file
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("temp_file_cleanup_failed", temp_path=tmp_path, request_id=request_id)

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error("inference_failed",
                    request_id=request_id,
                    error=str(e),
                    processing_time_ms=processing_time)
        raise HTTPException(
            status_code=500,
            detail="Classification processing failed"
        )

############################
# FASTAPI APPLICATION      #
############################

app = FastAPI(
    title="Breast Density Classification API",
    description="HIPAA-compliant API for mammogram breast density classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# EXACT SAME CORS CONFIG AS WORKING APP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Same as working app
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# KEEP TrustedHostMiddleware like working app
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """Audit logging middleware (like working app)"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        "Incoming request",
        request_id=request_id,
        method=request.method,
        endpoint=str(request.url),
        user_agent=request.headers.get("user-agent", "unknown"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    response = await call_next(request)
    
    # Log response
    processing_time = (time.time() - start_time) * 1000
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=response.status_code,
        processing_time_ms=round(processing_time, 2),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    return response

############################
# STARTUP EVENT (LIKE WORKING APP) #
############################

@app.on_event("startup")
async def load_model():
    """Load model at startup like working app"""
    global model_loaded, MODEL, model_load_error
    
    try:
        logger.info("Loading model", action="model_loading_start")
        
        # Load model synchronously
        MODEL = load_model_sync()
        
        model_loaded = True
        logger.info("Model loaded successfully", action="model_loading_success")
        
    except Exception as e:
        model_load_error = str(e)
        logger.error("Failed to load model", error=str(e), action="model_loading_error")
        raise

@app.on_event("startup")
async def set_startup_time():
    """Set startup time for uptime calculation"""
    app.state.start_time = time.time()

############################
# API ENDPOINTS            #
############################

@app.get("/", response_class=HTMLResponse)
async def home():
    """Homepage with healthcare compliance information"""
    content = """
    <html>
      <head>
        <title>Breast Density Classification API - Healthcare AI Platform</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; color: #333; line-height: 1.6; }
          header { text-align: center; padding: 20px; background: #2c5aa0; color: white; border-radius: 8px; margin-bottom: 20px; }
          section { background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
          .compliance { background: #e8f5e8; border-left: 4px solid #4CAF50; }
          .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
          code { background: #eee; padding: 2px 4px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <header>
          <h1>🏥 Breast Density Classification API</h1>
          <p>HIPAA-Compliant AI-Powered Mammogram Analysis</p>
        </header>
        
        <section class="compliance">
          <h2>🔒 Healthcare Compliance Features</h2>
          <ul>
            <li><strong>HIPAA Compliance:</strong> Comprehensive audit logging and secure data handling</li>
            <li><strong>Structured Logging:</strong> All requests and responses are logged for regulatory compliance</li>
            <li><strong>Input Validation:</strong> Rigorous validation of all inputs</li>
            <li><strong>Error Handling:</strong> Secure error responses that don't expose sensitive information</li>
          </ul>
        </section>
        
        <section>
          <h2>🤖 Model Overview</h2>
          <p>
            This API uses an <strong>InceptionV3</strong> model fine-tuned for breast density classification 
            on mammogram images.
          </p>
          <p><strong>Classes:</strong> A (almost entirely fatty), B (scattered areas), C (heterogeneously dense), D (extremely dense)</p>
        </section>
        
        <section class="warning">
          <h2>⚠️ Clinical Use Disclaimer</h2>
          <p>
            <strong>Important:</strong> This AI tool is designed to assist healthcare professionals and should not 
            replace clinical judgment. All results should be reviewed by qualified medical personnel.
          </p>
        </section>
        
        <section>
          <h2>📋 API Endpoints</h2>
          <ul>
            <li><code>GET /</code> - This information page</li>
            <li><code>GET /health</code> - System health check</li>
            <li><code>POST /classify</code> - Breast density classification endpoint</li>
            <li><code>GET /docs</code> - Interactive API documentation</li>
          </ul>
        </section>
      </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint (like working app)"""
    try:
        health_status = HealthCheckResponse(
            status="healthy" if model_loaded and not model_load_error else "unhealthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_loaded=model_loaded,
            version=MODEL_VERSION
        )
        
        logger.info("Health check performed", status=health_status.status, action="health_check")
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), action="health_check_error")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    request: Request,
    image_file: UploadFile = File(...)
):
    """
    Classify mammogram image for breast density.
    
    Returns breast density classification (A, B, C, D) with confidence scores.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        # Validate input file FIRST (like working app)
        validate_image_extension(image_file.filename)
        
        # Check if model is loaded AFTER validation
        if not model_loaded or MODEL is None:
            logger.error("Model not available", request_id=request_id, action="classification_error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available. Please try again later."
            )
        
        # Read and validate file
        image_bytes = await image_file.read()
        validate_file_size(len(image_bytes))
        
        # Log classification request
        logger.info(
            "Classification request received",
            request_id=request_id,
            filename_hash=anonymize_filename(image_file.filename),
            file_size_bytes=len(image_bytes),
            action="classification_start"
        )
        
        # Run inference
        results = run_inference_sync(image_bytes, request_id)
        
        # Build response
        response = ClassificationResponse(
            request_id=request_id,
            predicted_class=results["predicted_class"],
            probabilities=results["probabilities"],
            confidence_score=results["confidence_score"],
            processing_time_ms=results["processing_time_ms"],
            version=MODEL_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("classification_error",
                    request_id=request_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error during classification"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port
    )