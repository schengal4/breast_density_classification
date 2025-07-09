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
    """Application configuration using environment variables."""
    # Use the compatible model file
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model_compatible.pt")
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
    """Load model synchronously with proper compatibility handling"""
    try:
        # Log environment info for debugging
        logger.info("loading_model", 
                   pytorch_version=torch.__version__,
                   model_path=Config.MODEL_PATH,
                   device=str(DEVICE))
        
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

        # Load checkpoint with compatibility handling
        try:
            # Try with weights_only=False first (for newer PyTorch)
            checkpoint = torch.load(Config.MODEL_PATH, map_location=DEVICE, weights_only=False)
            logger.info("model_loaded_with_weights_only_false")
        except Exception as e:
            # Try loading to CPU first, then move to device
            logger.warning("retrying_with_cpu_load", error=str(e))
            checkpoint = torch.load(Config.MODEL_PATH, map_location='cpu', weights_only=False)
            logger.info("model_loaded_to_cpu")

        # Your checkpoint is directly an OrderedDict (state_dict)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            logger.info("extracted_state_dict_from_checkpoint")
        else:
            # Direct state_dict (which is your case)
            state_dict = checkpoint
            logger.info("using_direct_state_dict")

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning("missing_keys_in_model", 
                          count=len(missing_keys), 
                          sample_keys=missing_keys[:3])
        if unexpected_keys:
            logger.warning("unexpected_keys_in_model", 
                          count=len(unexpected_keys), 
                          sample_keys=unexpected_keys[:3])

        model.eval()
        logger.info("model_loaded_successfully", 
                   model_path=Config.MODEL_PATH, 
                   device=str(DEVICE))
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

# Replace your startup event with this crash-resistant version:
@app.on_event("startup")
async def load_model():
    """Load model without crashing the app if it fails"""
    global model_loaded, MODEL, model_load_error
    
    try:
        print("=== STARTUP: Attempting model load ===")
        
        # Critical: Don't let ANY exception crash the app
        if not os.path.exists(Config.MODEL_PATH):
            model_load_error = f"Model file missing: {Config.MODEL_PATH}"
            print(f"ERROR: {model_load_error}")
            model_loaded = False
            MODEL = None
            return  # Exit gracefully - app continues without model
        
        MODEL = load_model_sync()
        model_loaded = True
        print("=== SUCCESS: Model loaded ===")
        
    except Exception as e:
        # CRITICAL: Catch everything, never crash
        model_load_error = str(e)
        print(f"ERROR: Model failed to load: {model_load_error}")
        model_loaded = False
        MODEL = None
        # App continues running even with model failure
    
    print(f"Startup complete. Model loaded: {model_loaded}")

# Make health check always work (required for Health Universe)
@app.get("/health")
async def health_check():
    """Health check that NEVER fails"""
    try:
        status = "healthy" if model_loaded else "degraded"
        return {
            "status": status,
            "model_loaded": model_loaded,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    except:
        # Even if this fails, return something
        return {"status": "error", "timestamp": time.time()}
    
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

# Add these 3 debug endpoints:

@app.get("/ping")
async def ping():
    """Simple test - if this works, app is running"""
    return "OK"

@app.get("/debug")
async def debug_status():
    """See what's wrong with the model"""
    return {
        "app_running": True,
        "model_loaded": model_loaded,
        "model_error": model_load_error,
        "model_path": Config.MODEL_PATH,
        "model_file_exists": os.path.exists(Config.MODEL_PATH),
        "current_dir": os.getcwd()
    }

# @app.get("/files")
# async def debug_files():
#     """Check if model file was uploaded"""
#     try:
#         result = {"current_dir": os.getcwd()}
        
#         # Check models directory
#         if os.path.exists("models"):
#             result["models_dir"] = os.listdir("models")
#         else:
#             result["models_dir"] = "NOT_FOUND"
        
#         # Check root directory
#         result["root_files"] = os.listdir(".")[:10]  # First 10 files
        
#         return result
#     except Exception as e:
#         return {"error": str(e)}
    
# Add this enhanced debug endpoint to check file integrity:

# @app.get("/debug/file")
# async def debug_model_file():
#     """Check if model file is corrupted"""
#     try:
#         result = {}
        
#         if os.path.exists(Config.MODEL_PATH):
#             # Check file size
#             file_size = os.path.getsize(Config.MODEL_PATH)
#             result["file_size_bytes"] = file_size
#             result["file_size_mb"] = round(file_size / (1024*1024), 2)
            
#             # Check if file is too small (likely corrupted)
#             if file_size < 1024:  # Less than 1KB
#                 result["status"] = "CORRUPTED - File too small"
#                 result["first_bytes"] = "File too small to read"
#             else:
#                 # Read first 50 bytes to check file type
#                 with open(Config.MODEL_PATH, 'rb') as f:
#                     first_bytes = f.read(50)
                
#                 result["first_50_bytes"] = str(first_bytes)
#                 result["starts_with_pk"] = first_bytes.startswith(b'PK')  # ZIP file
#                 result["starts_with_pytorch"] = first_bytes.startswith(b'\x80\x03')  # PyTorch pickle
                
#                 # Check if it's a valid PyTorch file by trying to load metadata only
#                 try:
#                     with open(Config.MODEL_PATH, 'rb') as f:
#                         # Try to read the pickle header
#                         import pickle
#                         f.seek(0)
#                         header = f.read(10)
#                         result["pickle_header"] = str(header)
                        
#                         if len(header) >= 2:
#                             result["is_pickle_format"] = header[0:2] in [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']
                        
#                 except Exception as e:
#                     result["pickle_check_error"] = str(e)
                
#                 # Final assessment
#                 if file_size < 1024*1024:  # Less than 1MB is suspicious for a model
#                     result["status"] = "LIKELY_CORRUPTED - Too small for a model file"
#                 elif not (first_bytes.startswith(b'PK') or first_bytes.startswith(b'\x80')):
#                     result["status"] = "CORRUPTED - Not a valid PyTorch/ZIP file"
#                 else:
#                     result["status"] = "File appears valid"
#         else:
#             result["status"] = "FILE_NOT_FOUND"
        
#         return result
#     except Exception as e:
#         return {"error": f"Debug failed: {str(e)}"}
# Replace ONLY your main block with this:
if __name__ == "__main__":
    import uvicorn
    
    # CRITICAL: Health Universe expects port 8080
    port = int(os.getenv("PORT", 8080))  # Changed from 8000
    
    print(f"Starting on port {port} for Health Universe")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port
    )