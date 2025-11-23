"""
Hybrid Ovarian Cyst Classification System - FastAPI Backend
A clinical-grade web application for automatic ovarian cyst classification
using ultrasound images with ConvNeXt-Tiny hybrid CNN–Transformer model.
"""

import os
import uuid
import tempfile
import glob
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np

# Matplotlib imports - make optional for graceful degradation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Matplotlib not available: {e}")
    print("Server will start but visualization features will be disabled.")
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None
    plt = None

# TensorFlow imports - make optional for graceful degradation
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    TENSORFLOW_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} imported successfully")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    print(f"[WARNING] TensorFlow not available: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("Server will start but ML features will be disabled.")
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras_image = None

from PIL import Image

# scipy import - make optional (for advanced image analysis)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None
    print("Warning: scipy not available. Advanced image verification features will be limited.")

# SHAP import - make optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Explainability features will be limited.")
    SHAP_AVAILABLE = False
    shap = None
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, ForeignKey, func, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime as dt
from passlib.context import CryptContext
from jose import jwt, JWTError
import requests
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = "sqlite:///./users.db"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# FastAPI app
app = FastAPI(title="Ovarian Cyst Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/shap", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/insights", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# Database Models
# ============================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=True)
    agree_to_terms = Column(Boolean, default=False)
    google_id = Column(String, nullable=True)
    profile_pic = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    cases = relationship("PatientCase", back_populates="user")
    batches = relationship("Batch", back_populates="user")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)
    patient_id = Column(String, unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=True)
    clinical_notes = Column(String, nullable=True)
    date_of_scan = Column(String, nullable=False)
    
    # Relationships
    cases = relationship("PatientCase", back_populates="patient")

class PatientCase(Base):
    __tablename__ = "patient_cases"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=True, index=True)
    image_path = Column(String, nullable=False)
    shap_path = Column(String, nullable=True)
    verification_score = Column(Float, nullable=True)  # Changed from Boolean to Float (0.0-1.0)
    prob_simple = Column(Float, nullable=True)  # Changed from String to Float
    prob_complex = Column(Float, nullable=True)  # Changed from String to Float
    prediction_label = Column(Integer, nullable=True)  # 0=Simple, 1=Complex
    predicted_class = Column(String, nullable=True)  # Kept for backward compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="cases")
    patient = relationship("Patient", back_populates="cases")
    batch = relationship("Batch", back_populates="cases")
    annotation = relationship("Annotation", back_populates="case", uselist=False)

class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("patient_cases.id"), index=True)
    radiologist_name = Column(String, nullable=False)
    comments = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    follow_up = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    case = relationship("PatientCase", back_populates="annotation")

class Batch(Base):
    __tablename__ = "batches"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    batch_name = Column(String, nullable=True)
    status = Column(String, default="uploading")  # uploading, uploaded, processing, completed, failed, cancelled
    total_cases = Column(Integer, default=0)
    completed_cases = Column(Integer, default=0)
    failed_cases = Column(Integer, default=0)
    pending_cases = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="batches")
    cases = relationship("PatientCase", back_populates="batch")

# Create all tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# Pydantic Models
# ============================================================================

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    agreeToTerms: bool

class LoginRequest(BaseModel):
    email: str
    password: str
    rememberMe: bool = False

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class GoogleAuthRequest(BaseModel):
    credential: str

class PatientCaseRequest(BaseModel):
    patient_name: str
    patient_id: str
    age: int
    gender: str
    date_of_scan: str
    symptoms: str

class AnnotationRequest(BaseModel):
    case_id: int
    radiologist_name: str
    comments: str
    severity: str
    follow_up: str

class UploadResponse(BaseModel):
    case_id: int
    verification_score: float
    verification_passed: bool
    message: str

# ============================================================================
# Database Dependency
# ============================================================================

def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# Authentication Utilities
# ============================================================================

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    """Create a JWT access token."""
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user(authorization: Optional[str] = Header(None, alias="Authorization"), db: Session = Depends(get_db)) -> User:
    """Get current authenticated user from JWT token."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    try:
        token = authorization.split(" ")[1] if " " in authorization else authorization
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        
        # Update last login
        user.last_login = datetime.utcnow()  # type: ignore[assignment]
        db.commit()
        
        return user
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

def get_current_user_from_request(request: Request, db: Session = Depends(get_db)) -> User:
    """Get current authenticated user from JWT token using Request object (for multipart/form-data)."""
    # Try to get authorization header (case-insensitive)
    authorization = None
    # FastAPI's request.headers is case-insensitive, but let's be explicit
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header:
        authorization = auth_header
    
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    try:
        token = authorization.split(" ")[1] if " " in authorization else authorization
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        
        # Update last login
        user.last_login = datetime.utcnow()  # type: ignore[assignment]
        db.commit()
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

# ============================================================================
# ML Model Loading
# ============================================================================

# ML Model Loading - only if TensorFlow is available
model = None
per_image_norm: Optional[callable] = None  # type: ignore[assignment]

if TENSORFLOW_AVAILABLE and tf is not None:
    try:
        from keras.saving import register_keras_serializable
    except (ImportError, ModuleNotFoundError):
        # Fallback for TensorFlow 2.x
        try:
            from tensorflow.keras.saving import register_keras_serializable
        except (ImportError, ModuleNotFoundError, AttributeError):
            register_keras_serializable = None
            print("Warning: Could not import register_keras_serializable")
    
    if register_keras_serializable and tf is not None:
        # Define the custom normalization function
        def per_image_norm(x):
            """Custom normalization layer."""
            return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)  # type: ignore[attr-defined]
        
        # Register it with Keras - this must be done before loading the model
        try:
            register_keras_serializable(package="Custom")(per_image_norm)
        except Exception as reg_e:
            print(f"Warning: Could not register per_image_norm: {reg_e}")
        
        # Load ConvNeXt-Tiny model
        MODEL_PATH = os.path.join("models", "convnext_tiny_ovarian", "convnext_tiny_baseline.keras")
        try:
            # Pass the function in custom_objects - required for loading
            model = tf.keras.models.load_model(  # type: ignore[attr-defined]
                MODEL_PATH,
                custom_objects={"per_image_norm": per_image_norm}
            )
            print(f"[OK] Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"[WARNING] Could not load model from {MODEL_PATH}: {e}")
            import traceback
            traceback.print_exc()
            model = None
else:
    print("Warning: TensorFlow not available. Model cannot be loaded.")

# ============================================================================
# Ultrasound Verification
# ============================================================================

def verify_ultrasound(img: Image.Image) -> float:
    """
    Verify if image appears to be a valid ultrasound image.
    Returns a score between 0.0 and 1.0.
    Score >= 0.65 indicates likely valid ultrasound.
    
    Balanced validation: More lenient for actual ultrasound images,
    but still rejects obvious non-medical images (photos, colorful images).
    """
    try:
        # Convert to RGB if needed, then to grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        grayscale = img.convert('L')
        arr = np.array(grayscale, dtype=np.float32)
        color_arr = np.array(img, dtype=np.float32)
        
        # Basic statistics
        mean_intensity = np.mean(arr)
        std_intensity = np.std(arr)
        min_intensity = np.min(arr)
        max_intensity = np.max(arr)
        
        # Score calculation - start with base score and adjust
        score = 0.5  # Start with neutral score (more lenient)
        
        # 1. Brightness check (ultrasound images vary widely, be lenient)
        if 50 <= mean_intensity <= 200:
            score += 0.15  # Good range
        elif 30 <= mean_intensity < 50 or 200 < mean_intensity <= 230:
            score += 0.05  # Acceptable range
        elif mean_intensity < 20 or mean_intensity > 240:
            score -= 0.3  # Only penalize extreme values (likely not ultrasound)
        
        # 2. Contrast check (ultrasound has varying contrast, be lenient)
        if 15 <= std_intensity <= 70:
            score += 0.15  # Good range
        elif 10 <= std_intensity < 15 or 70 < std_intensity <= 85:
            score += 0.05  # Acceptable range
        elif std_intensity < 5 or std_intensity > 100:
            score -= 0.2  # Only penalize extreme values
        
        # 3. Color saturation check - THIS IS THE KEY: reject colorful images
        if color_arr.ndim == 3 and color_arr.shape[2] == 3:
            try:
                # Calculate color channel differences
                r_g_diff = np.mean(np.abs(color_arr[..., 0] - color_arr[..., 1]))
                r_b_diff = np.mean(np.abs(color_arr[..., 0] - color_arr[..., 2]))
                g_b_diff = np.mean(np.abs(color_arr[..., 1] - color_arr[..., 2]))
                avg_color_diff = (r_g_diff + r_b_diff + g_b_diff) / 3.0
                
                # Ultrasound images should be mostly grayscale
                if avg_color_diff < 8:
                    score += 0.2  # Very grayscale (good for ultrasound)
                elif avg_color_diff < 15:
                    score += 0.1  # Mostly grayscale (acceptable)
                elif avg_color_diff > 30:
                    score -= 0.4  # High color saturation (likely photo, not ultrasound)
                elif avg_color_diff > 50:
                    score -= 0.6  # Very colorful (definitely not ultrasound)
            except Exception as e:
                print(f"Color analysis error: {e}")
                # If color analysis fails, don't penalize - just skip this check
        
        # 4. Dynamic range check (ultrasound typically has good range)
        dynamic_range = max_intensity - min_intensity
        if dynamic_range > 80:  # Good dynamic range
            score += 0.1
        elif dynamic_range < 30:  # Very limited range (might be issue)
            score -= 0.1
        
        # 5. Edge detection - reject images with very sharp edges (photos)
        if SCIPY_AVAILABLE and ndimage is not None:
            try:
                sobel_x = ndimage.sobel(arr, axis=1)
                sobel_y = ndimage.sobel(arr, axis=0)
                edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_mean = np.mean(edge_magnitude)
                
                # Very sharp edges suggest photos, not ultrasound
                if edge_mean > 120:
                    score -= 0.3  # Very sharp edges (likely photo)
                elif edge_mean > 150:
                    score -= 0.5  # Extremely sharp (definitely photo)
                # Don't penalize moderate or low edge strength (ultrasound varies)
            except Exception as e:
                print(f"Edge detection error: {e}")
                # Edge detection failed, skip this check
                pass
        
        # 6. Texture analysis - reject images with very high texture (photos)
        if SCIPY_AVAILABLE and ndimage is not None:
            try:
                kernel = np.ones((5, 5)) / 25.0
                local_mean = ndimage.convolve(arr, kernel)
                local_variance = ndimage.convolve((arr - local_mean)**2, kernel)
                avg_local_variance = np.mean(local_variance)
                
                # Very high variance suggests photos with lots of detail
                if avg_local_variance > 8000:
                    score -= 0.2  # Very high texture (likely photo)
            except Exception as e:
                print(f"Texture analysis error: {e}")
                # Texture analysis failed, skip this check
                pass
        
        # Ensure score stays in valid range
        score = max(0.0, min(1.0, score))
        
        print(f"DEBUG: Verification score = {score:.3f} (mean={mean_intensity:.1f}, std={std_intensity:.1f}, range={dynamic_range:.1f})")
        
        return score
    except Exception as e:
        print(f"Verification error: {e}")
        import traceback
        traceback.print_exc()
        # Return a more lenient score on error to avoid rejecting valid images
        return 0.6  # Return slightly above threshold to allow through with review

# ============================================================================
# Explainability Functions
# ============================================================================

def explain_by_occlusion(model, img_array, patch=64):
    """
    Generate occlusion-based explainability heatmap.
    Uses larger patch size (64) by default for faster processing (~4x faster than patch=32).
    """
    if not TENSORFLOW_AVAILABLE or model is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))
    try:
        base_pred = model.predict(img_array, verbose=0)[0][0]
        h, w, _ = img_array.shape[1:]
        heatmap = np.zeros((h, w))
        for y in range(0, h, patch):
            for x in range(0, w, patch):
                occluded = np.array(img_array, copy=True)
                occluded[0, y:y+patch, x:x+patch, :] = 0
                pred = model.predict(occluded, verbose=0)[0][0]
                importance = base_pred - pred
                heatmap[y:y+patch, x:x+patch] = importance
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-9)
        return heatmap
    except Exception as e:
        print(f"Occlusion explainability error: {e}")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

def generate_shap_explanation(model, img_array):
    """
    Generate SHAP explainability visualization using GradientExplainer.
    Optimized for ConvNeXt-Tiny model.
    """
    if not SHAP_AVAILABLE or not TENSORFLOW_AVAILABLE or model is None or shap is None:
        return None
    try:
        # Use GradientExplainer for deep learning models
        # This is more efficient than DeepExplainer for modern architectures
        explainer = shap.GradientExplainer(model, img_array)
        
        # Calculate SHAP values - this may take 1-5 minutes depending on hardware
        shap_values = explainer.shap_values(img_array)
        
        # Extract SHAP values for the predicted class
        # For binary classification, shap_values is typically a list with one element
        if isinstance(shap_values, list):
            shap_img = shap_values[0][0]  # Get first (and only) class, first sample
        else:
            shap_img = shap_values[0]  # Single array case
        
        # Normalize SHAP values to [0, 1] for visualization
        if hasattr(shap_img, 'min') and hasattr(shap_img, 'max'):
            shap_min = shap_img.min()  # type: ignore[attr-defined]
            shap_max = shap_img.max()  # type: ignore[attr-defined]
            shap_range = shap_max - shap_min + 1e-9
            shap_img = (shap_img - shap_min) / shap_range  # type: ignore[assignment]
        else:
            shap_img = np.array(shap_img)
            shap_min = np.min(shap_img)
            shap_max = np.max(shap_img)
            shap_range = shap_max - shap_min + 1e-9
            shap_img = (shap_img - shap_min) / shap_range
        
        # Take absolute values to show importance (both positive and negative contributions)
        shap_img = np.abs(shap_img)
        
        return shap_img
    except Exception as e:
        print(f"SHAP explainability error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/signup", response_model=dict)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """User registration endpoint."""
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        name=request.name,
        email=request.email,
        password=get_password_hash(request.password),
        agree_to_terms=request.agreeToTerms
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": user.email})
    return {
        "message": "Signup successful",
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email}
    }

@app.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User login endpoint."""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not user.password or not verify_password(request.password, str(user.password)):  # type: ignore[truthy-function]
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user.last_login = datetime.utcnow()  # type: ignore[assignment]
    db.commit()
    
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email, "profile_pic": user.profile_pic}
    }

@app.post("/google-auth", response_model=TokenResponse)
def google_auth(request: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Google OAuth authentication endpoint."""
    google_token_info_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={request.credential}"
    resp = requests.get(google_token_info_url)
    
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    
    info = resp.json()
    if info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Invalid Google client ID")
    
    email = info.get("email")
    google_id = info.get("sub")
    name = info.get("name", "")
    picture = info.get("picture", "")
    
    # Find or create user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, name=name, google_id=google_id, profile_pic=picture)
        db.add(user)
    else:
        # Update profile picture if available
        if picture and not user.profile_pic:  # type: ignore[truthy-function]
            user.profile_pic = picture  # type: ignore[assignment]
        user.last_login = datetime.utcnow()  # type: ignore[assignment]
    
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email, "profile_pic": user.profile_pic}
    }

# ============================================================================
# Profile Endpoints
# ============================================================================

@app.get("/profile")
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get current user profile."""
    case_count = db.query(PatientCase).filter(PatientCase.user_id == current_user.id).count()
    
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "profile_pic": current_user.profile_pic,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,  # type: ignore[truthy-function]
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,  # type: ignore[truthy-function]
        "cases_analyzed": case_count
    }

@app.put("/profile")
def update_profile(
    name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    if name:
        current_user.name = name  # type: ignore[assignment]
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated", "user": {"id": current_user.id, "name": current_user.name, "email": current_user.email}}

# ============================================================================
# Upload & Prediction Endpoints
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_case(
    patient_name: str = Form(...),
    patient_id: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    date_of_scan: str = Form(...),
    symptoms: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload ultrasound image with patient details.
    Returns case_id and verification score.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    image_path = None
    try:
        # Save uploaded file
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename or "")[1] or ".png"
        image_filename = f"{patient_id}_{date_of_scan}_{uuid.uuid4().hex}{file_ext}"
        image_path = os.path.join(upload_dir, image_filename)
        
        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and verify image
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB format
            verification_score = verify_ultrasound(img)
            
            # New verification thresholds:
            # < 0.50: Reject (not ultrasound)
            # 0.50-0.65: Review required (user must confirm)
            # >= 0.65: Auto-approve (continue)
            verification_status = "rejected"
            if verification_score < 0.50:
                verification_status = "rejected"
            elif verification_score < 0.65:
                verification_status = "review_required"
            else:
                verification_status = "approved"
            
            # Reject image if verification score is below 0.50
            if verification_status == "rejected":
                # Clean up uploaded file
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image verification failed. Verification score: {verification_score:.3f} (minimum: 0.50). The uploaded image does not appear to be a valid medical ultrasound image. Please upload a proper ultrasound scan image."
                )
        except HTTPException:
            # Re-raise HTTP exceptions (verification failures)
            raise
        except Exception as e:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Find or create patient
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            patient = Patient(
                patient_name=patient_name,
                patient_id=patient_id,
                age=age,
                gender=gender,
                clinical_notes=symptoms,
                date_of_scan=date_of_scan
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)
        
        # Create case record
        case = PatientCase(
            user_id=current_user.id,
            patient_id=patient.id,
            image_path=image_path,
            verification_score=verification_score
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        
        return {
            "case_id": case.id,
            "verification_score": verification_score,
            "verification_status": verification_status,
            "verification_passed": verification_status == "approved",  # For backward compatibility
            "message": "Image uploaded successfully. Use /predict/{case_id} to get classification." if verification_status == "approved" else "Image uploaded. Review required before prediction."
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Clean up uploaded file if case creation fails
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass  # Ignore file removal errors during cleanup
        print(f"Error in upload_case: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# Batch Processing Endpoints
# ============================================================================

@app.post("/batch/upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...),  # JSON string with array of patient data
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload multiple ultrasound images with patient details.
    metadata should be JSON array: [{"patient_name": "...", "patient_id": "...", "age": ..., "gender": "...", "date_of_scan": "...", "symptoms": "..."}, ...]
    """
    
    try:
        import json
        metadata_list = json.loads(metadata)
        
        if len(files) != len(metadata_list):
            raise HTTPException(status_code=400, detail="Number of files must match number of metadata entries")
        
        # Create batch record
        batch = Batch(
            user_id=current_user.id,
            batch_name=f"Batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            status="uploading",
            total_cases=len(files)
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        
        uploaded_cases = []
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        for idx, (file, meta) in enumerate(zip(files, metadata_list)):
            image_path = None
            try:
                # Save file
                file_ext = os.path.splitext(file.filename or "")[1] or ".png"
                image_filename = f"{meta['patient_id']}_{meta['date_of_scan']}_{uuid.uuid4().hex}{file_ext}"
                image_path = os.path.join(upload_dir, image_filename)
                
                content = await file.read()
                with open(image_path, "wb") as buffer:
                    buffer.write(content)
                
                # Verify image
                img = Image.open(image_path)
                img = img.convert('RGB')
                verification_score = verify_ultrasound(img)
                
                verification_status = "rejected"
                if verification_score < 0.50:
                    verification_status = "rejected"
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)  # Clean up rejected file
                    uploaded_cases.append({
                        "index": idx,
                        "filename": file.filename,
                        "status": "rejected",
                        "verification_score": verification_score,
                        "error": "Verification failed (score < 0.50)"
                    })
                    batch.failed_cases += 1  # type: ignore[assignment]
                    continue
                elif verification_score < 0.65:
                    verification_status = "review_required"
                    batch.pending_cases += 1  # type: ignore[assignment]
                else:
                    verification_status = "approved"
                
                # Find or create patient
                patient = db.query(Patient).filter(Patient.patient_id == meta['patient_id']).first()
                if not patient:
                    patient = Patient(
                        patient_name=meta['patient_name'],
                        patient_id=meta['patient_id'],
                        age=meta['age'],
                        gender=meta.get('gender', ''),
                        clinical_notes=meta.get('symptoms', ''),
                        date_of_scan=meta['date_of_scan']
                    )
                    db.add(patient)
                    db.commit()
                    db.refresh(patient)
                
                # Create case
                case = PatientCase(
                    user_id=current_user.id,
                    patient_id=patient.id,
                    image_path=image_path,
                    verification_score=verification_score,
                    batch_id=batch.id
                )
                db.add(case)
                db.commit()
                db.refresh(case)
                
                uploaded_cases.append({
                    "index": idx,
                    "case_id": case.id,
                    "filename": file.filename,
                    "status": verification_status,
                    "verification_score": verification_score
                })
                
            except Exception as e:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass
                uploaded_cases.append({
                    "index": idx,
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
                batch.failed_cases += 1  # type: ignore[assignment]
        
        batch.status = "uploaded"  # type: ignore[assignment]
        db.commit()
        
        return {
            "batch_id": batch.id,
            "batch_name": batch.batch_name,
            "total_files": len(files),
            "uploaded_cases": uploaded_cases,
            "summary": {
                "approved": sum(1 for c in uploaded_cases if c.get("status") == "approved"),
                "review_required": sum(1 for c in uploaded_cases if c.get("status") == "review_required"),
                "rejected": sum(1 for c in uploaded_cases if c.get("status") == "rejected"),
                "errors": sum(1 for c in uploaded_cases if c.get("status") == "error")
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

@app.post("/batch/predict/{batch_id}")
async def batch_predict(
    batch_id: int,
    auto_approve_review: bool = Query(False, description="Auto-approve cases requiring review"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start batch prediction processing.
    Returns SSE stream with progress updates.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get all cases in batch that can be processed
    query = db.query(PatientCase).filter(PatientCase.batch_id == batch_id)
    if auto_approve_review:
        query = query.filter(PatientCase.verification_score >= 0.50)
    else:
        query = query.filter(PatientCase.verification_score >= 0.65)
    query = query.filter(PatientCase.predicted_class.is_(None))  # Not yet predicted
    
    cases = query.all()
    
    if not cases:
        raise HTTPException(status_code=400, detail="No cases available for prediction")
    
    batch.status = "processing"  # type: ignore[assignment]
    batch.started_at = datetime.utcnow()  # type: ignore[assignment]
    batch.total_cases = len(cases)  # type: ignore[assignment]
    db.commit()
    
    async def generate_progress():
        completed = 0
        failed = 0
        
        for case in cases:
            try:
                data = json.dumps({"status": "progress", "case_id": case.id, "message": f"Processing case {completed + failed + 1}/{len(cases)}"})
                yield f"data: {data}\n\n"
                
                # Run prediction (reuse existing function)
                case_id_val = int(case.id)  # type: ignore[arg-type]
                for update in run_prediction_with_progress(case_id_val, False, current_user, db):
                    if update.get("status") == "complete":
                        completed += 1
                        batch.completed_cases = completed  # type: ignore[assignment]
                        db.commit()
                        data = json.dumps({
                            "status": "case_complete",
                            "case_id": case.id,
                            "progress": int(((completed + failed) / len(cases)) * 100),
                            "message": f"Completed {completed}/{len(cases)} cases"
                        })
                        yield f"data: {data}\n\n"
                        break
                    elif update.get("status") == "error":
                        failed += 1
                        batch.failed_cases = failed  # type: ignore[assignment]
                        db.commit()
                        data = json.dumps({
                            "status": "case_error",
                            "case_id": case.id,
                            "error": update.get("message", "Unknown error")
                        })
                        yield f"data: {data}\n\n"
                        break
                    else:
                        data = json.dumps({
                            "status": "progress",
                            "case_id": case.id,
                            "message": update.get("message", "")
                        })
                        yield f"data: {data}\n\n"
                        
            except Exception as e:
                failed += 1
                batch.failed_cases = failed  # type: ignore[assignment]
                db.commit()
                data = json.dumps({
                    "status": "case_error",
                    "case_id": case.id,
                    "error": str(e)
                })
                yield f"data: {data}\n\n"
        
        batch.status = "completed"  # type: ignore[assignment]
        batch.completed_at = datetime.utcnow()  # type: ignore[assignment]
        db.commit()
        
        final_data = json.dumps({
            "status": "complete",
            "message": f"Batch processing complete. {completed} succeeded, {failed} failed.",
            "summary": {
                "total": len(cases),
                "completed": completed,
                "failed": failed
            }
        })
        yield f"data: {final_data}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/batch/{batch_id}")
def get_batch_status(
    batch_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get batch status and all cases."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized")
    
    cases = db.query(PatientCase).filter(PatientCase.batch_id == batch_id).all()
    
    case_details = []
    for case in cases:
        verification_score_val = case.verification_score if case.verification_score is not None else None
        predicted_class_val = case.predicted_class if case.predicted_class is not None else None
        
        # Determine verification status
        if verification_score_val is not None:
            if verification_score_val >= 0.65:  # type: ignore[operator]
                verification_status = "approved"
            elif verification_score_val >= 0.50:  # type: ignore[operator]
                verification_status = "review_required"
            else:
                verification_status = "rejected"
        else:
            verification_status = "rejected"
        
        case_details.append({
            "case_id": case.id,
            "patient_name": case.patient.patient_name if case.patient else "Unknown",
            "verification_score": verification_score_val,
            "verification_status": verification_status,
            "predicted_class": predicted_class_val,
            "prob_simple": case.prob_simple,
            "prob_complex": case.prob_complex,
            "status": "completed" if predicted_class_val is not None else "pending"  # type: ignore[operator]
        })
    
    return {
        "batch_id": batch.id,
        "batch_name": batch.batch_name,
        "status": batch.status,
        "total_cases": batch.total_cases,
        "completed_cases": batch.completed_cases,
        "failed_cases": batch.failed_cases,
        "pending_cases": batch.pending_cases,
        "created_at": batch.created_at.isoformat() if batch.created_at is not None else None,
        "started_at": batch.started_at.isoformat() if batch.started_at is not None else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at is not None else None,
        "cases": case_details
    }

@app.get("/batches")
def list_batches(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all batches for current user."""
    batches = db.query(Batch).filter(Batch.user_id == current_user.id)\
        .order_by(Batch.created_at.desc()).offset(skip).limit(limit).all()
    total = db.query(Batch).filter(Batch.user_id == current_user.id).count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "batches": [{
            "batch_id": b.id,
            "batch_name": b.batch_name,
            "status": b.status,
            "total_cases": b.total_cases,
            "completed_cases": b.completed_cases,
            "failed_cases": b.failed_cases,
            "created_at": b.created_at.isoformat() if b.created_at is not None else None
        } for b in batches]
    }

def run_prediction_with_progress(case_id: int, include_shap: bool, current_user: User, db: Session):
    """
    Helper function that runs prediction and yields progress updates.
    Yields dictionaries with 'status' and 'message' keys.
    Note: This is a regular generator (not async) because the operations are blocking.
    """
    try:
        yield {"status": "loading", "message": "Loading and preprocessing image..."}
        
        if not model:
            yield {"status": "error", "message": "Model not loaded"}
            return
        
        case = db.query(PatientCase).filter(PatientCase.id == case_id).first()
        if not case:
            yield {"status": "error", "message": "Case not found"}
            return
        
        if case.user_id != current_user.id:  # type: ignore[operator]
            yield {"status": "error", "message": "Not authorized to access this case"}
            return
        
        verification_score_val = case.verification_score  # type: ignore[assignment]
        if verification_score_val is not None:
            score = float(verification_score_val)  # type: ignore[arg-type]  # SQLAlchemy resolves to Python value at runtime
            if score < 0.50:
                yield {"status": "error", "message": f"Image verification failed (score: {score:.3f}). Image does not appear to be a valid ultrasound. Please upload a new image."}
                return
            elif score < 0.65:
                yield {"status": "error", "message": f"Image verification requires review (score: {score:.3f}). Please confirm this is a valid ultrasound image before proceeding."}
                return
        
        if not TENSORFLOW_AVAILABLE or keras_image is None:
            yield {"status": "error", "message": "TensorFlow is not available"}
            return
        
        # Load and preprocess image
        image_path_str = str(case.image_path) if case.image_path is not None else None
        if not image_path_str or not os.path.exists(image_path_str):
            yield {"status": "error", "message": "Image file not found"}
            return
        
        img = keras_image.load_img(image_path_str, target_size=(256, 256))
        arr = keras_image.img_to_array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        yield {"status": "progress", "message": "Image loaded and preprocessed"}
        
        # Prediction
        yield {"status": "progress", "message": "Running model prediction..."}
        
        # Single-value sigmoid model: raw_pred = [[p_simple]]
        raw_pred = model.predict(arr, verbose=0)
        print(f"DEBUG: Raw model output shape: {raw_pred.shape}, values: {raw_pred}")
        
        raw_val = float(raw_pred[0][0])
        
        # Clamp to valid probability range
        p_simple = max(0.0, min(1.0, raw_val))
        p_complex = 1.0 - p_simple
        
        # Final class
        if p_simple >= 0.5:
            pred_class = "Simple/Benign Cyst"
            prediction_label = 0
        else:
            pred_class = "Complex/Malignant Cyst"
            prediction_label = 1
        
        confidence = round(max(p_simple, p_complex), 3)
        
        print(f"DEBUG: p_simple={p_simple:.4f}, p_complex={p_complex:.4f}, "
              f"pred_class={pred_class}, confidence={confidence}")
        yield {"status": "progress", "message": f"Prediction complete: {pred_class} (confidence: {confidence:.3f})"}
        
        # Generate explainability visualizations
        yield {"status": "progress", "message": "Generating explainability visualizations..."}
        shap_dir = "static/shap"
        os.makedirs(shap_dir, exist_ok=True)
        
        # Occlusion heatmap
        heatmap_filename = None
        overlay_filename = None
        try:
            yield {"status": "progress", "message": "Generating occlusion heatmap (this may take 10-30 seconds)..."}
            occlusion_heatmap = explain_by_occlusion(model, arr, patch=64)
            heatmap_filename = f"occlusion_heatmap_{case_id}_{uuid.uuid4().hex}.png"
            overlay_filename = f"occlusion_overlay_{case_id}_{uuid.uuid4().hex}.png"
            heatmap_path = os.path.join(shap_dir, heatmap_filename)
            overlay_path = os.path.join(shap_dir, overlay_filename)
            
            if MATPLOTLIB_AVAILABLE and plt is not None:
                plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                plt.imshow(occlusion_heatmap, cmap="jet")  # type: ignore[attr-defined]
                plt.axis("off")  # type: ignore[attr-defined]
                plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                plt.close()  # type: ignore[attr-defined]
                
                plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                plt.imshow(img)  # type: ignore[attr-defined]
                plt.imshow(occlusion_heatmap, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
                plt.axis("off")  # type: ignore[attr-defined]
                plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                plt.close()  # type: ignore[attr-defined]
                yield {"status": "progress", "message": "[OK] Occlusion heatmap saved"}
        except Exception as e:
            yield {"status": "warning", "message": f"Error generating occlusion heatmap: {str(e)}"}
        
        # SHAP explainability
        shap_heatmap_filename = None
        shap_overlay_filename = None
        shap_overlay_path = None
        if include_shap and SHAP_AVAILABLE:
            try:
                yield {"status": "progress", "message": "Generating SHAP explanation (this may take 1-5 minutes)..."}
                shap_img = generate_shap_explanation(model, arr)
                
                if shap_img is not None and MATPLOTLIB_AVAILABLE and plt is not None:
                    shap_heatmap_filename = f"shap_heatmap_{case_id}_{uuid.uuid4().hex}.png"
                    shap_overlay_filename = f"shap_overlay_{case_id}_{uuid.uuid4().hex}.png"
                    shap_heatmap_path = os.path.join(shap_dir, shap_heatmap_filename)
                    shap_overlay_path = os.path.join(shap_dir, shap_overlay_filename)
                    
                    plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                    plt.imshow(shap_img, cmap="jet")  # type: ignore[attr-defined]
                    plt.axis('off')  # type: ignore[attr-defined]
                    plt.savefig(shap_heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                    plt.close()  # type: ignore[attr-defined]
                    
                    plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                    plt.imshow(img)  # type: ignore[attr-defined]
                    plt.imshow(shap_img, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
                    plt.axis('off')  # type: ignore[attr-defined]
                    plt.savefig(shap_overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                    plt.close()  # type: ignore[attr-defined]
                    yield {"status": "progress", "message": "[OK] SHAP heatmap saved"}
                    
                    if not overlay_filename:
                        case.shap_path = shap_overlay_path  # type: ignore[assignment]
            except Exception as e:
                yield {"status": "warning", "message": f"Error generating SHAP explanation: {str(e)}"}
        
        # Update case record
        yield {"status": "progress", "message": "Updating case record in database..."}
        case.prob_simple = p_simple  # type: ignore[assignment]
        case.prob_complex = p_complex  # type: ignore[assignment]
        case.prediction_label = prediction_label  # type: ignore[assignment]
        case.predicted_class = pred_class  # type: ignore[assignment]
        if shap_overlay_filename:
            case.shap_path = shap_overlay_path  # type: ignore[assignment]
        elif overlay_filename:
            case.shap_path = overlay_path  # type: ignore[assignment]
        else:
            case.shap_path = None  # type: ignore[assignment]
        db.commit()
        
        # Final result
        result = {
            "status": "complete",
            "message": "Prediction complete!",
            "data": {
                "case_id": case_id,
                "predicted_class": pred_class,
                "prediction_label": prediction_label,
                "probabilities": {
                    "Simple": p_simple,
                    "Complex": p_complex
                },
                "confidence": confidence,
                "verification_score": case.verification_score,
                "occlusion_heatmap": heatmap_filename,
                "occlusion_overlay": overlay_filename,
                "shap_heatmap": shap_heatmap_filename,
                "shap_overlay": shap_overlay_filename,
                "explanation": "Highlighted areas contributed most to the prediction."
            }
        }
        yield result
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"status": "error", "message": f"Prediction failed: {str(e)}"}

@app.get("/predict/{case_id}/stream")
async def predict_case_stream(
    case_id: int,
    include_shap: bool = Query(False, description="Generate SHAP explainability"),
    token: Optional[str] = Query(None, description="JWT token (required if not in Authorization header)"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Stream prediction progress updates using Server-Sent Events (SSE).
    The frontend can use EventSource to receive real-time progress updates.
    Token can be passed as query parameter (for EventSource) or Authorization header (for fetch).
    """
    # Authenticate user - support both query param (for EventSource) and header (for fetch)
    auth_token = None
    if token:
        auth_token = token
    elif authorization:
        auth_token = authorization.split(" ")[1] if " " in authorization else authorization
    else:
        async def event_generator_error():
            error_data = json.dumps({"status": "error", "message": "Authentication required"})
            yield f"data: {error_data}\n\n"
        return StreamingResponse(
            event_generator_error(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    payload = verify_token(auth_token)
    if not payload:
        async def event_generator_error():
            error_data = json.dumps({"status": "error", "message": "Invalid token"})
            yield f"data: {error_data}\n\n"
        return StreamingResponse(
            event_generator_error(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    email = payload.get("sub")
    if not email:
        async def event_generator_error():
            error_data = json.dumps({"status": "error", "message": "Invalid token"})
            yield f"data: {error_data}\n\n"
        return StreamingResponse(
            event_generator_error(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    current_user = db.query(User).filter(User.email == email).first()
    if not current_user:
        async def event_generator_error():
            error_data = json.dumps({"status": "error", "message": "User not found"})
            yield f"data: {error_data}\n\n"
        return StreamingResponse(
            event_generator_error(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    async def event_generator():
        try:
            # Run blocking prediction - iterate through generator
            for progress in run_prediction_with_progress(case_id, include_shap, current_user, db):
                # Format as SSE message
                data = json.dumps(progress)
                yield f"data: {data}\n\n"
                # Small delay to allow events to be sent
                await asyncio.sleep(0.01)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"status": "error", "message": f"Stream error: {str(e)}"})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/predict/{case_id}")
async def predict_case(
    case_id: int, 
    include_shap: bool = Query(False, description="Generate SHAP explainability (disabled by default, may take 1-5 minutes)"),
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """
    Run prediction and explainability on an uploaded case.
    include_shap: Whether to generate SHAP explainability (default: False for faster processing, may take 1-5 minutes if enabled)
    Note: After prediction, radiologist can add comments via /annotate endpoint before generating report.
    """
    print(f"Starting prediction for case_id: {case_id}")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    case = db.query(PatientCase).filter(PatientCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Check ownership
    if case.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized to access this case")
    
    # Check verification score before processing
    verification_score_val = case.verification_score  # type: ignore[assignment]
    if verification_score_val is not None:
        score = float(verification_score_val)  # type: ignore[arg-type]  # SQLAlchemy resolves to Python value at runtime
        if score < 0.50:
            raise HTTPException(
                status_code=400,
                detail=f"Image verification failed (score: {score:.3f}). Image does not appear to be a valid ultrasound. Please upload a new image."
            )
        elif score < 0.65:
            raise HTTPException(
                status_code=400,
                detail=f"Image verification requires review (score: {score:.3f}). Please confirm this is a valid ultrasound image before proceeding."
            )
    
    # Load and preprocess image
    if not TENSORFLOW_AVAILABLE or keras_image is None:
        raise HTTPException(status_code=503, detail="TensorFlow is not available. Please install TensorFlow to use ML features.")
    try:
        image_path_str = str(case.image_path) if case.image_path is not None else None
        if not image_path_str or not os.path.exists(image_path_str):
            raise HTTPException(status_code=404, detail="Image file not found")
        print(f"Loading image from: {image_path_str}")
        # Preprocess image exactly as done in Kaggle training:
        # 1. Load image
        # 2. Resize to 256 × 256
        # 3. Convert to float32
        # 4. Normalize to [0,1]
        # 5. Expand dimensions to (1, 256, 256, 3)
        img = keras_image.load_img(image_path_str, target_size=(256, 256))
        arr = keras_image.img_to_array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        print("Image loaded and preprocessed (256x256, float32, normalized [0,1], batch dimension added)")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
    
    # Prediction
    print("Running model prediction...")
    try:
        # Single-value sigmoid model: raw_pred = [[p_simple]]
        raw_pred = model.predict(arr, verbose=0)
        print(f"DEBUG: Raw model output shape: {raw_pred.shape}, values: {raw_pred}")
        
        raw_val = float(raw_pred[0][0])
        
        # Clamp to valid probability range
        p_simple = max(0.0, min(1.0, raw_val))
        p_complex = 1.0 - p_simple
        
        # Final class
        if p_simple >= 0.5:
            pred_class = "Simple/Benign Cyst"
            prediction_label = 0
        else:
            pred_class = "Complex/Malignant Cyst"
            prediction_label = 1
        
        confidence = round(max(p_simple, p_complex), 3)
        
        print(f"DEBUG: p_simple={p_simple:.4f}, p_complex={p_complex:.4f}, "
              f"pred_class={pred_class}, confidence={confidence}")
        print(f"Prediction complete: {pred_class} (confidence: {confidence})")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Generate explainability visualizations (optional, can be slow)
    print("Generating explainability visualizations...")
    shap_dir = "static/shap"
    os.makedirs(shap_dir, exist_ok=True)
    
    # Occlusion heatmap (faster than SHAP)
    heatmap_filename = None
    overlay_filename = None
    try:
        print("Generating occlusion heatmap (this may take 10-30 seconds)...")
        occlusion_heatmap = explain_by_occlusion(model, arr, patch=64)  # Larger patch for faster processing
        heatmap_filename = f"occlusion_heatmap_{case_id}_{uuid.uuid4().hex}.png"
        overlay_filename = f"occlusion_overlay_{case_id}_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(shap_dir, heatmap_filename)
        overlay_path = os.path.join(shap_dir, overlay_filename)
        
        if MATPLOTLIB_AVAILABLE and plt is not None:
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(occlusion_heatmap, cmap="jet")  # type: ignore[attr-defined]
            plt.axis("off")  # type: ignore[attr-defined]
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
            
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(img)  # type: ignore[attr-defined]
            plt.imshow(occlusion_heatmap, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
            plt.axis("off")  # type: ignore[attr-defined]
            plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
            print("[OK] Occlusion heatmap saved")
        else:
            print("Matplotlib not available, skipping occlusion heatmap")
    except Exception as e:
        print(f"[ERROR] Error generating occlusion heatmap: {str(e)}")
        import traceback
        traceback.print_exc()
        # Continue without occlusion heatmap
    
    # SHAP explainability (enabled by default, can be slow - 1-5 minutes)
    shap_heatmap_filename = None
    shap_overlay_filename = None
    shap_overlay_path = None
    if include_shap and SHAP_AVAILABLE:
        try:
            print("Generating SHAP explanation (this may take 1-5 minutes)...")
            shap_img = generate_shap_explanation(model, arr)
            
            if shap_img is not None and MATPLOTLIB_AVAILABLE and plt is not None:
                shap_heatmap_filename = f"shap_heatmap_{case_id}_{uuid.uuid4().hex}.png"
                shap_overlay_filename = f"shap_overlay_{case_id}_{uuid.uuid4().hex}.png"
                shap_heatmap_path = os.path.join(shap_dir, shap_heatmap_filename)
                shap_overlay_path = os.path.join(shap_dir, shap_overlay_filename)
                
                plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                plt.imshow(shap_img, cmap="jet")  # type: ignore[attr-defined]
                plt.axis('off')  # type: ignore[attr-defined]
                plt.savefig(shap_heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                plt.close()  # type: ignore[attr-defined]
                
                plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
                plt.imshow(img)  # type: ignore[attr-defined]
                plt.imshow(shap_img, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
                plt.axis('off')  # type: ignore[attr-defined]
                plt.savefig(shap_overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
                plt.close()  # type: ignore[attr-defined]
                print("[OK] SHAP heatmap saved")
                
                # Update case with SHAP overlay path if occlusion overlay doesn't exist
                if not overlay_filename:
                    case.shap_path = shap_overlay_path  # type: ignore[assignment]
            else:
                print("SHAP image not available or matplotlib not available")
        except Exception as e:
            print(f"[ERROR] Error generating SHAP explanation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without SHAP - occlusion heatmap is still available
    elif not SHAP_AVAILABLE:
        print("SHAP not available - skipping SHAP explainability")
    
    # Update case record
    print("Updating case record in database...")
    case.prob_simple = p_simple  # type: ignore[assignment]
    case.prob_complex = p_complex  # type: ignore[assignment]
    case.prediction_label = prediction_label  # type: ignore[assignment]
    case.predicted_class = pred_class  # type: ignore[assignment]
    # Prefer SHAP overlay if available, otherwise use occlusion overlay
    if shap_overlay_filename:
        case.shap_path = shap_overlay_path  # type: ignore[assignment]
    elif overlay_filename:
        case.shap_path = overlay_path  # type: ignore[assignment]
    else:
        case.shap_path = None  # type: ignore[assignment]
    db.commit()
    print("Case record updated")
    
    result = {
        "case_id": case_id,
        "predicted_class": pred_class,
        "prediction_label": prediction_label,
        "probabilities": {
            "Simple": p_simple,
            "Complex": p_complex
        },
        "confidence": confidence,
        "verification_score": case.verification_score,
        "occlusion_heatmap": heatmap_filename,
        "occlusion_overlay": overlay_filename,
        "shap_heatmap": shap_heatmap_filename,
        "shap_overlay": shap_overlay_filename,
        "explanation": "Highlighted areas contributed most to the prediction."
    }
    print(f"Returning prediction result for case_id: {case_id}")
    return result

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    """
    Quick prediction endpoint (for testing, no case creation).
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Load and preprocess
        if not TENSORFLOW_AVAILABLE or keras_image is None:
            os.remove(tmp_path)
            raise HTTPException(status_code=503, detail="TensorFlow is not available. Please install TensorFlow to use ML features.")
        # Preprocess image exactly as done in Kaggle training
        img = keras_image.load_img(tmp_path, target_size=(256, 256))
        arr = keras_image.img_to_array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Verification
        verification_score = verify_ultrasound(img)
        
        # Prediction
        # Single-value sigmoid model: raw_pred = [[p_simple]]
        raw_pred = model.predict(arr, verbose=0)
        
        raw_val = float(raw_pred[0][0])
        
        # Clamp to valid probability range
        p_simple = max(0.0, min(1.0, raw_val))
        p_complex = 1.0 - p_simple
        
        # Final class
        if p_simple >= 0.5:
            pred_class = "Simple/Benign Cyst"
            prediction_label = 0
        else:
            pred_class = "Complex/Malignant Cyst"
            prediction_label = 1
        
        confidence = round(max(p_simple, p_complex), 3)
        
        print(f"DEBUG: p_simple={p_simple:.4f}, p_complex={p_complex:.4f}, "
              f"pred_class={pred_class}, confidence={confidence}")
        print(f"Prediction complete: {pred_class} (confidence: {confidence})")
        
        # Generate explainability
        shap_dir = "static/shap"
        os.makedirs(shap_dir, exist_ok=True)
        
        occlusion_heatmap = explain_by_occlusion(model, arr)
        heatmap_filename = f"temp_heatmap_{uuid.uuid4().hex}.png"
        overlay_filename = f"temp_overlay_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(shap_dir, heatmap_filename)
        overlay_path = os.path.join(shap_dir, overlay_filename)
        
        if MATPLOTLIB_AVAILABLE and plt is not None:
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(occlusion_heatmap, cmap="jet")  # type: ignore[attr-defined]
            plt.axis("off")  # type: ignore[attr-defined]
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
            
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(img)  # type: ignore[attr-defined]
            plt.imshow(occlusion_heatmap, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
            plt.axis("off")  # type: ignore[attr-defined]
            plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
        
        # SHAP
        shap_heatmap_filename = None
        shap_overlay_filename = None
        shap_img = generate_shap_explanation(model, arr)
        
        if shap_img is not None and MATPLOTLIB_AVAILABLE and plt is not None:
            shap_heatmap_filename = f"temp_shap_heatmap_{uuid.uuid4().hex}.png"
            shap_overlay_filename = f"temp_shap_overlay_{uuid.uuid4().hex}.png"
            shap_heatmap_path = os.path.join(shap_dir, shap_heatmap_filename)
            shap_overlay_path = os.path.join(shap_dir, shap_overlay_filename)
            
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(shap_img, cmap="jet")  # type: ignore[attr-defined]
            plt.axis('off')  # type: ignore[attr-defined]
            plt.savefig(shap_heatmap_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
            
            plt.figure(figsize=(4, 4))  # type: ignore[attr-defined]
            plt.imshow(img)  # type: ignore[attr-defined]
            plt.imshow(shap_img, cmap="jet", alpha=0.45)  # type: ignore[attr-defined]
            plt.axis('off')  # type: ignore[attr-defined]
            plt.savefig(shap_overlay_path, bbox_inches='tight', pad_inches=0)  # type: ignore[attr-defined]
            plt.close()  # type: ignore[attr-defined]
        
        return {
            "predicted_class": pred_class,
            "probabilities": {
                "Simple": p_simple,
                "Complex": p_complex
            },
            "confidence": confidence,
            "verification_score": verification_score,
            "verification_passed": verification_score >= 0.65,
            "occlusion_heatmap": heatmap_filename,
            "occlusion_overlay": overlay_filename,
            "shap_heatmap": shap_heatmap_filename,
            "shap_overlay": shap_overlay_filename
        }
    finally:
        os.remove(tmp_path)

# ============================================================================
# Case Management Endpoints
# ============================================================================

@app.get("/cases")
def get_cases(
    skip: int = Query(0, ge=0, description="Number of cases to skip"),
    limit: int = Query(20, ge=1, description="Number of cases per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's case history with pagination, ordered by confidence (highest first)."""
    # Get ALL cases for the user with eager loading of relationships
    all_cases = db.query(PatientCase).options(
        joinedload(PatientCase.patient),
        joinedload(PatientCase.annotation)
    ).filter(
        PatientCase.user_id == current_user.id
    ).all()
    
    # Calculate confidence for each case
    cases_with_confidence = []
    for c in all_cases:
        # Calculate confidence if probabilities exist
        if c.prob_simple is not None and c.prob_complex is not None:
            prob_simple_val = float(c.prob_simple)  # type: ignore[arg-type]
            prob_complex_val = float(c.prob_complex)  # type: ignore[arg-type]
            confidence = max(prob_simple_val, prob_complex_val)
        else:
            # Pending cases get confidence 0 (will appear at the end)
            confidence = 0.0
        cases_with_confidence.append((c, confidence))
    
    # Sort by confidence (highest first), then by created_at for same confidence
    # Use a very old date for None created_at to put them at the end
    min_date = datetime(1900, 1, 1)
    cases_with_confidence.sort(key=lambda x: (x[1], x[0].created_at if x[0].created_at else min_date), reverse=True)
    
    # Apply pagination
    total = len(cases_with_confidence)
    paginated_cases = cases_with_confidence[skip:skip + limit]
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "cases": [{
            "id": c.id,
            "patient_name": c.patient.patient_name if c.patient else "Unknown",
            "age": c.patient.age if c.patient else None,
            "gender": c.patient.gender if c.patient else None,
            "date_of_scan": c.patient.date_of_scan if c.patient else None,
            "predicted_class": c.predicted_class,
            "prediction_label": c.prediction_label,
            "verification_score": c.verification_score,
            "prob_simple": c.prob_simple,
            "prob_complex": c.prob_complex,
            "confidence": round(confidence * 100, 1),
            "created_at": c.created_at.isoformat() if c.created_at else None,  # type: ignore[truthy-function]
            "has_annotation": c.annotation is not None,
            "annotation_preview": c.annotation.comments[:50] + "..." if c.annotation and c.annotation.comments else None  # type: ignore[truthy-function]
        } for c, confidence in paginated_cases]
    }

@app.get("/dashboard/stats")
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics for the current user."""
    # Total cases
    total_cases = db.query(PatientCase).filter(PatientCase.user_id == current_user.id).count()
    
    # Cases by classification
    simple_cases = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.prediction_label == 0
    ).count()
    
    complex_cases = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.prediction_label == 1
    ).count()
    
    pending_cases = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.predicted_class.is_(None)
    ).count()
    
    # Cases this month
    first_day_this_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cases_this_month = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.created_at >= first_day_this_month
    ).count()
    
    # Average confidence score: average of top 5 highest confidences
    cases_with_probs = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.prob_simple.isnot(None),
        PatientCase.prob_complex.isnot(None)
    ).all()
    
    if cases_with_probs:
        confidences = []
        for c in cases_with_probs:
            prob_simple_val = float(c.prob_simple) if c.prob_simple is not None else 0.0  # type: ignore[arg-type]
            prob_complex_val = float(c.prob_complex) if c.prob_complex is not None else 0.0  # type: ignore[arg-type]
            confidences.append(max(prob_simple_val, prob_complex_val))
        
        # Get top 5 highest confidences
        if len(confidences) >= 5:
            top_5_confidences = sorted(confidences, reverse=True)[:5]
            avg_confidence = float(sum(top_5_confidences) / len(top_5_confidences) * 100)
        else:
            # If less than 5 cases, use all available
            avg_confidence = float(sum(confidences) / len(confidences) * 100) if confidences else 0.0
    else:
        avg_confidence = 0.0
    
    # Recent cases: top 7 cases by confidence score
    recent_cases = []
    if cases_with_probs:
        # Calculate confidence for each case and sort
        cases_with_confidence = []
        for c in cases_with_probs:
            prob_simple_val = float(c.prob_simple) if c.prob_simple is not None else 0.0  # type: ignore[arg-type]
            prob_complex_val = float(c.prob_complex) if c.prob_complex is not None else 0.0  # type: ignore[arg-type]
            confidence = max(prob_simple_val, prob_complex_val)
            cases_with_confidence.append((c, confidence))
        
        # Sort by confidence (highest first) and take top 7
        cases_with_confidence.sort(key=lambda x: x[1], reverse=True)
        top_7_cases = cases_with_confidence[:7]
        
        for case, confidence in top_7_cases:
            recent_cases.append({
                "case_id": case.id,
                "patient_name": case.patient.patient_name if case.patient else "Unknown",
                "predicted_class": case.predicted_class,
                "confidence": round(confidence * 100, 1),
                "created_at": case.created_at.isoformat() if case.created_at else None
            })
    
    # Cases this week
    first_day_this_week = datetime.now() - timedelta(days=datetime.now().weekday())
    first_day_this_week = first_day_this_week.replace(hour=0, minute=0, second=0, microsecond=0)
    cases_this_week = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id,
        PatientCase.created_at >= first_day_this_week
    ).count()
    
    # Last case date
    last_case = db.query(PatientCase).filter(
        PatientCase.user_id == current_user.id
    ).order_by(PatientCase.created_at.desc()).first()
    
    last_case_date = None
    if last_case is not None and last_case.created_at is not None:
        last_case_date = last_case.created_at.isoformat()
    
    return {
        "total_cases": total_cases,
        "simple_cases": simple_cases,
        "complex_cases": complex_cases,
        "pending_cases": pending_cases,
        "cases_this_month": cases_this_month,
        "cases_this_week": cases_this_week,
        "avg_confidence": round(avg_confidence, 1),
        "last_case_date": last_case_date,
        "recent_cases": recent_cases
    }

@app.get("/case/{case_id}")
def get_case(case_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get detailed case information."""
    case = db.query(PatientCase).filter(PatientCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    if case.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized to access this case")
    
    annotation = case.annotation
    patient = case.patient
    
    return {
        "case": {
            "id": case.id,
            "patient_name": patient.patient_name if patient else "Unknown",
            "patient_id": patient.patient_id if patient else None,
            "age": patient.age if patient else None,
            "gender": patient.gender if patient else None,
            "date_of_scan": patient.date_of_scan if patient else None,
            "symptoms": patient.clinical_notes if patient else None,
            "image_path": case.image_path,
            "verification_score": case.verification_score,
            "verification_passed": case.verification_score >= 0.65 if case.verification_score is not None else False,  # type: ignore[operator]
            "predicted_class": case.predicted_class,
            "prediction_label": case.prediction_label,
            "prob_simple": case.prob_simple,
            "prob_complex": case.prob_complex,
            "shap_path": case.shap_path,
            "created_at": case.created_at.isoformat() if case.created_at else None  # type: ignore[truthy-function]
        },
        "annotation": {
            "radiologist_name": annotation.radiologist_name if annotation else None,
            "comments": annotation.comments if annotation else None,
            "severity": annotation.severity if annotation else None,
            "follow_up": annotation.follow_up if annotation else None,
            "created_at": annotation.created_at.isoformat() if annotation and annotation.created_at else None  # type: ignore[truthy-function]
        } if annotation else None
    }

# ============================================================================
# Annotation Endpoints
# ============================================================================

@app.post("/annotate")
async def annotate(request: AnnotationRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Add radiologist annotation to a case."""
    case = db.query(PatientCase).filter(PatientCase.id == request.case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    if case.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized to annotate this case")
    
    # Check if annotation already exists
    existing = db.query(Annotation).filter(Annotation.case_id == request.case_id).first()
    if existing:  # type: ignore[truthy-function]
        # Update existing annotation
        existing.radiologist_name = request.radiologist_name  # type: ignore[assignment]
        existing.comments = request.comments  # type: ignore[assignment]
        existing.severity = request.severity  # type: ignore[assignment]
        existing.follow_up = request.follow_up  # type: ignore[assignment]
        annotation = existing
    else:
        # Create new annotation
        annotation = Annotation(
            case_id=request.case_id,
            radiologist_name=request.radiologist_name,
            comments=request.comments,
            severity=request.severity,
            follow_up=request.follow_up
        )
        db.add(annotation)
    
    db.commit()
    db.refresh(annotation)
    return {"annotation_id": annotation.id, "message": "Annotation saved successfully"}

# ============================================================================
# Report Generation
# ============================================================================

@app.get("/report/{case_id}")
async def generate_report(
    case_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """
    Generate PDF report for a case with images.
    Note: Report can be generated after prediction is complete.
    Radiologist comments can be added via /annotate endpoint before generating report.
    """
    case = db.query(PatientCase).filter(PatientCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    if case.user_id != current_user.id:  # type: ignore[operator]
        raise HTTPException(status_code=403, detail="Not authorized to access this case")
    
    # Ensure prediction has been run
    if case.predicted_class is None:
        raise HTTPException(
            status_code=400,
            detail="Prediction has not been run for this case. Please run prediction first via /predict/{case_id}"
        )
    
    annotation = case.annotation
    patient = case.patient
    
    pdf_dir = "static/reports"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"report_{case_id}.pdf")
    
    # Find explainability images
    shap_dir = "static/shap"
    shap_path_str = str(case.shap_path) if case.shap_path is not None else None
    occlusion_overlay_path = shap_path_str if shap_path_str and os.path.exists(shap_path_str) else None
    
    # Find occlusion heatmap by searching for files matching the pattern
    occlusion_heatmap_path = None
    shap_heatmap_path = None
    shap_overlay_path = None
    if os.path.exists(shap_dir):
        # Find occlusion heatmap
        heatmap_pattern = os.path.join(shap_dir, f"occlusion_heatmap_{case_id}_*.png")
        heatmap_files = glob.glob(heatmap_pattern)
        if heatmap_files:
            occlusion_heatmap_path = heatmap_files[0]  # Use the first match
        
        # Find SHAP heatmap and overlay
        shap_heatmap_pattern = os.path.join(shap_dir, f"shap_heatmap_{case_id}_*.png")
        shap_heatmap_files = glob.glob(shap_heatmap_pattern)
        if shap_heatmap_files:
            shap_heatmap_path = shap_heatmap_files[0]
        
        shap_overlay_pattern = os.path.join(shap_dir, f"shap_overlay_{case_id}_*.png")
        shap_overlay_files = glob.glob(shap_overlay_pattern)
        if shap_overlay_files:
            shap_overlay_path = shap_overlay_files[0]
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Color scheme matching frontend
    # #047857 (green), #374151 (dark gray), #6b7280 (medium gray), #f3f4f6 (light gray), #e5e7eb (border)
    green = (0.016, 0.471, 0.341)  # #047857
    dark_gray = (0.216, 0.255, 0.318)  # #374151
    medium_gray = (0.420, 0.455, 0.502)  # #6b7280
    light_gray = (0.953, 0.957, 0.965)  # #f3f4f6
    border_gray = (0.898, 0.906, 0.922)  # #e5e7eb
    red = (0.863, 0.149, 0.149)  # #dc2626
    
    # Title with green color
    c.setFillColor(green)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Ovarian Cyst Classification Report")
    
    y_pos = height - 90
    c.setFont("Helvetica", 10)
    c.setFillColor(medium_gray)
    c.drawString(50, y_pos, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    y_pos -= 30
    
    # Patient Information Section
    c.setFillColor(dark_gray)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Patient Information")
    # Draw border line
    c.setStrokeColor(border_gray)
    c.setLineWidth(2)
    c.line(50, y_pos - 5, width - 50, y_pos - 5)
    y_pos -= 30
    c.setFont("Helvetica", 11)
    
    if patient:
        # Grid layout for patient info (2 columns with better alignment)
        col1_x = 50
        col2_x = width / 2 + 15  # Better column spacing
        label_width = 120  # Fixed width for labels to ensure alignment
        value_indent = 15  # Indent for values
        
        # Starting Y position for both columns
        start_y = y_pos
        row_height = 30  # Consistent row height
        current_row = 0
        
        # Helper function to draw a label-value pair
        def draw_field(col_x, row, label, value, value_color=None, value_bold=False):
            field_y = start_y - (row * row_height)
            # Label
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(col_x, field_y, label)
            # Value
            if value_color:
                c.setFillColor(value_color)
            else:
                c.setFillColor(dark_gray)
            if value_bold:
                c.setFont("Helvetica-Bold", 10)
            else:
                c.setFont("Helvetica", 10)
            c.drawString(col_x + label_width, field_y, str(value))
        
        # Column 1 - Left side
        draw_field(col1_x, current_row, "Patient Name:", patient.patient_name)
        current_row += 1
        
        draw_field(col1_x, current_row, "Patient ID:", patient.patient_id)
        current_row += 1
        
        draw_field(col1_x, current_row, "Age:", str(patient.age))
        current_row += 1
        
        # Column 2 - Right side (reset row counter)
        current_row = 0
        patient_gender = patient.gender  # type: ignore[assignment]
        if patient_gender:
            draw_field(col2_x, current_row, "Gender:", str(patient_gender))
            current_row += 1
        
        draw_field(col2_x, current_row, "Date of Scan:", patient.date_of_scan)
        current_row += 1
        
        # Verification Score with color coding
        verification_score_val = case.verification_score  # type: ignore[assignment]
        if verification_score_val is not None:
            score = float(verification_score_val)  # type: ignore[arg-type]  # SQLAlchemy resolves to Python value at runtime
            score_color = green if score >= 0.65 else red
            draw_field(col2_x, current_row, "Verification Score:", f"{score:.3f}", value_color=score_color, value_bold=True)
            if score < 0.65:
                c.setFillColor(red)
                c.setFont("Helvetica", 9)
                c.drawString(col2_x + label_width, start_y - (current_row * row_height) - 12, "(Below threshold)")
        else:
            draw_field(col2_x, current_row, "Verification Score:", "N/A")
        
        # Update y_pos to below the highest column
        max_rows = max(3, current_row + 1)  # At least 3 rows from column 1
        y_pos = start_y - (max_rows * row_height) - 10
        
        if patient.clinical_notes:
            # Clinical Notes in a box
            notes = patient.clinical_notes
            y_pos -= 10  # Extra spacing before clinical notes
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_pos, "Clinical Notes:")
            y_pos -= 25
            
            # Calculate box height based on content
            words = notes.split()
            lines = []
            current_line = ""
            max_width = width - 110  # Account for margins
            chars_per_line = int(max_width / 6)  # Approximate chars per line (6 points per char)
            
            for word in words:
                if len(current_line + word) < chars_per_line:
                    current_line += word + " "
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            
            num_lines = min(len(lines), 4)  # Max 4 lines
            box_height = (num_lines * 15) + 10
            box_y = y_pos - box_height
            
            # Draw background box for clinical notes
            c.setFillColor(light_gray)
            c.rect(50, box_y, width - 100, box_height, fill=1, stroke=0)
            
            # Text in box
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            for i, line in enumerate(lines[:num_lines]):
                c.drawString(55, box_y + box_height - 15 - (i * 15), line)
            
            y_pos = box_y - 10
    
    y_pos -= 10
    
    # Classification Results Section
    y_pos -= 15  # Extra spacing before section
    c.setFillColor(dark_gray)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Classification Results")
    # Draw border line
    c.setStrokeColor(border_gray)
    c.setLineWidth(2)
    c.line(50, y_pos - 5, width - 50, y_pos - 5)
    y_pos -= 35
    
    # Classification results in boxes (grid layout with better spacing)
    gap = 12  # Gap between boxes
    total_gap = gap * 2  # Gaps on left and right
    available_width = width - 100 - total_gap  # Account for margins and gaps
    box_width = available_width / 3
    box_height = 60  # Slightly taller for better readability
    box_y = y_pos - box_height
    
    # Helper function to draw a classification box
    def draw_classification_box(x, label, value, value_color=None, value_bold=True):
        # Background box
        c.setFillColor(light_gray)
        c.rect(x, box_y, box_width, box_height, fill=1, stroke=0)
        # Label
        c.setFillColor(medium_gray)
        c.setFont("Helvetica", 9)
        c.drawString(x + 8, box_y + box_height - 18, label)
        # Value
        if value_color:
            c.setFillColor(value_color)
        else:
            c.setFillColor(dark_gray)
        if value_bold:
            c.setFont("Helvetica-Bold", 13)
        else:
            c.setFont("Helvetica", 13)
        # Center the value vertically and horizontally
        value_y = box_y + 15
        c.drawString(x + 8, value_y, str(value))
    
    # Predicted Class box
    predicted_class_str = str(case.predicted_class) if case.predicted_class else 'N/A'  # type: ignore[assignment]
    draw_classification_box(50, "Predicted Class", predicted_class_str, value_color=green)
    
    # P(Simple) box
    prob_simple_val = case.prob_simple  # type: ignore[assignment]
    simple_value = f"{(prob_simple_val * 100):.1f}%" if prob_simple_val is not None else "N/A"
    draw_classification_box(50 + box_width + gap, "P(Simple)", simple_value)
    
    # P(Complex) box
    prob_complex_val = case.prob_complex  # type: ignore[assignment]
    complex_value = f"{(prob_complex_val * 100):.1f}%" if prob_complex_val is not None else "N/A"
    draw_classification_box(50 + (box_width + gap) * 2, "P(Complex)", complex_value)
    
    y_pos = box_y - 30
    
    # Images Section
    c.setFillColor(dark_gray)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Images")
    # Draw border line
    c.setStrokeColor(border_gray)
    c.setLineWidth(2)
    c.line(50, y_pos - 5, width - 50, y_pos - 5)
    y_pos -= 30
    
    # Images in grid layout (side by side if space allows)
    image_path_str = str(case.image_path) if case.image_path is not None else None
    img_size = 200  # Size in points (slightly larger for better visibility)
    img_gap = 20  # Gap between images
    img1_x = 50  # First image X position
    
    # Original ultrasound image
    img1_y_pos = y_pos  # Track Y position for alignment
    img1_height = 0
    if image_path_str and os.path.exists(image_path_str):
        try:
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(img1_x, y_pos, "Original Ultrasound")
            y_pos -= 8
            img = Image.open(image_path_str)
            img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
            img_width, img_height = img.size
            img_width_pt = img_width * 0.72
            img_height_pt = img_height * 0.72
            img1_height = img_height_pt  # Store for alignment
            # Draw border around image with rounded corners effect
            c.setStrokeColor(border_gray)
            c.setLineWidth(1.5)
            c.rect(img1_x, y_pos - img_height_pt - 3, img_width_pt + 6, img_height_pt + 6, fill=0, stroke=1)
            c.drawImage(ImageReader(img), img1_x + 3, y_pos - img_height_pt, width=img_width_pt, height=img_height_pt)
        except Exception as e:
            print(f"Error adding original image to PDF: {e}")
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            c.drawString(img1_x, y_pos, "Original image unavailable")
    
    # Explainability visualization (prefer SHAP overlay, fallback to occlusion)
    explainability_path = shap_overlay_path if shap_overlay_path and os.path.exists(shap_overlay_path) else occlusion_overlay_path
    if explainability_path and os.path.exists(explainability_path):
        try:
            # Calculate position for second image
            img2_x = img1_x + img_size * 0.72 + img_gap
            can_side_by_side = img2_x + img_size * 0.72 < width - 50 and img1_height > 0
            
            if can_side_by_side:
                # Side-by-side layout - align with first image
                c.setFillColor(medium_gray)
                c.setFont("Helvetica-Bold", 10)
                c.drawString(img2_x, img1_y_pos, "Explainability Visualization")
                img_y = img1_y_pos - 8 - img1_height  # Align bottom with first image
            else:
                # Stacked layout
                if img1_height > 0:
                    y_pos = y_pos - img1_height - 30  # Move down from first image
                c.setFillColor(medium_gray)
                c.setFont("Helvetica-Bold", 10)
                c.drawString(img1_x, y_pos, "Explainability Visualization")
                y_pos -= 8
                img2_x = img1_x  # Use same X position
                img_y = y_pos
            
            img = Image.open(explainability_path)
            img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
            img_width, img_height = img.size
            img_width_pt = img_width * 0.72
            img_height_pt = img_height * 0.72
            
            # Draw border around image
            c.setStrokeColor(border_gray)
            c.setLineWidth(1.5)
            c.rect(img2_x, img_y - img_height_pt - 3, img_width_pt + 6, img_height_pt + 6, fill=0, stroke=1)
            c.drawImage(ImageReader(img), img2_x + 3, img_y - img_height_pt, width=img_width_pt, height=img_height_pt)
            
            # Update y_pos for next section
            if can_side_by_side:
                # Side-by-side: move down from the taller image
                y_pos = min(img1_y_pos - 8 - img1_height, img_y - img_height_pt) - 30
            else:
                # Stacked: move down from second image
                y_pos = img_y - img_height_pt - 30
        except Exception as e:
            print(f"Error adding explainability image to PDF: {e}")
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            c.drawString(50, y_pos, "Explainability visualization unavailable")
            if img1_height > 0:
                y_pos = y_pos - img1_height - 30
            else:
                y_pos -= 20
    
    
    # Check if we need a new page
    if y_pos < 150:
        c.showPage()
        y_pos = height - 50
    
    # Annotation Section
    if annotation:
        # Check if we need a new page
        if y_pos < 200:
            c.showPage()
            y_pos = height - 50
        
        y_pos -= 10
        c.setFillColor(dark_gray)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Radiologist Annotation")
        # Draw border line
        c.setStrokeColor(border_gray)
        c.setLineWidth(2)
        c.line(50, y_pos - 5, width - 50, y_pos - 5)
        y_pos -= 30
        
        # Annotation in a box with light gray background
        annotation_box_height = 120
        c.setFillColor(light_gray)
        c.rect(50, y_pos - annotation_box_height, width - 100, annotation_box_height, fill=1, stroke=0)
        
        annotation_y = y_pos - 20
        
        # Radiologist name
        c.setFillColor(medium_gray)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(55, annotation_y, "Radiologist:")
        c.setFillColor(dark_gray)
        c.setFont("Helvetica", 10)
        c.drawString(55, annotation_y - 15, annotation.radiologist_name)
        annotation_y -= 35
        
        # Comments
        if annotation.comments:
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(55, annotation_y, "Comments:")
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            comments = annotation.comments
            words = comments.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 90:
                    current_line += word + " "
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                c.drawString(55, annotation_y - 15 - (i * 15), line)
            annotation_y -= (min(len(lines), 3) * 15) + 10
        
        # Severity
        if annotation.severity:
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(55, annotation_y, "Severity:")
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            c.drawString(55, annotation_y - 15, annotation.severity)
            annotation_y -= 30
        
        # Follow-up
        if annotation.follow_up:
            c.setFillColor(medium_gray)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(55, annotation_y, "Follow-up:")
            c.setFillColor(dark_gray)
            c.setFont("Helvetica", 10)
            c.drawString(55, annotation_y - 15, annotation.follow_up)
        
        y_pos -= annotation_box_height + 10
    
    # Disclaimer
    y_pos -= 20
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, y_pos, "Model Disclaimer: For clinical decision support only. Not a replacement for expert diagnosis.")
    y_pos -= 15
    c.drawString(50, y_pos, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if case.created_at:  # type: ignore[truthy-function]
        c.drawString(50, y_pos - 15, f"Case Created: {case.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    # Add clinician name
    if current_user and current_user.name:  # type: ignore[truthy-function]
        c.drawString(50, y_pos - 30, f"Clinician: {current_user.name}")  # type: ignore[attr-defined]
    
    c.save()
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"report_{case_id}.pdf")

# ============================================================================
# Insights Dashboard
# ============================================================================

@app.get("/insights")
async def get_insights():
    """Get comprehensive model performance metrics and insights."""
    return {
        "metrics": {
            "accuracy": 0.9506,
            "roc_auc": 0.9890,
            "precision_macro": 0.9409,
            "recall_macro": 0.9598,
            "f1_macro": 0.9502,
            "per_class": {
                "Complex": {
                    "precision": 0.96,
                    "recall": 0.94,
                    "f1": 0.95,
                    "support": 206
                },
                "Simple": {
                    "precision": 0.94,
                    "recall": 0.95,
                    "f1": 0.95,
                    "support": 199
                }
            },
            "confusion_matrix": [[194, 12], [9, 190]],
            "misclassified": 21,
            "test_size": 405
        },
        "dataset": {
            "train_complex": 957,
            "train_simple": 928,
            "train_total": 1885,
            "test_complex": 206,
            "test_simple": 199,
            "test_total": 405
        },
        "model_info": {
            "architecture": "ConvNeXt-Tiny + custom classification head",
            "input_shape": [256, 256, 3],
            "total_params": 28021345,
            "trainable_params": 199169,
            "non_trainable_params": 27822176,
            "loss": "BinaryCrossentropy (label_smoothing=0.05)",
            "optimizer": "Adam",
            "metrics": ["accuracy", "AUC (roc_auc)"],
            "preprocessing": "Per-image normalization: (x - mean) / (std + 1e-6)",
            "data_augmentation": [
                "RandomFlip(horizontal)",
                "RandomRotation(0.05)",
                "RandomZoom(0.10)",
                "RandomContrast(0.10)",
                "GaussianNoise(0.02)"
            ]
        },
        "explainability": {
            "methods": [
                "Occlusion-based sensitivity maps (deployed)",
                "SHAP (GradientExplainer, offline research)"
            ],
            "occlusion": {
                "description": "Image divided into patches, each patch occluded to measure drop in predicted probability",
                "patch_size": "32×32"
            },
            "shap": {
                "description": "GradientExplainer used to produce per-pixel contribution maps",
                "usage": "Research and documentation evidence"
            }
        },
        "verification": {
            "type": "Heuristic ultrasound verification",
            "score_range": [0.0, 1.0],
            "threshold": 0.65,
            "description": "Checks overall darkness, contrast patterns, and presence of characteristic fan-shaped region"
        },
        "clinical": {
            "intended_use": "Clinical decision support only",
            "limitations": "Not a replacement for expert diagnosis. For clinical decision support only.",
            "disclaimer": "This model is intended to assist healthcare professionals in ovarian cyst classification. All predictions should be reviewed by qualified medical professionals."
        },
        "charts": {
            "confusion_matrix": "/static/insights/confusion_matrix.png",
            "roc_curve": "/static/insights/roc_auc_curve.png",
            "pr_curve": "/static/insights/precision_recall_curve.png",
            "train_val_loss": "/static/insights/train_loss_curve.png",
            "train_val_roc_auc": "/static/insights/roc_auc_curve.png",
            "hist_simple": "/static/insights/simple_intensity_histogram.png",
            "hist_complex": "/static/insights/complex_intensity_histogram.png",
            "shap_simple": "/static/insights/shap_example_simple.png",
            "shap_complex": "/static/insights/shap_example_complex.png",
            "occlusion_simple": "/static/insights/occlusion_example_simple.png",
            "occlusion_complex": "/static/insights/occlusion_example_complex.png"
        }
    }

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/favicon.ico")
def favicon():
    """Return 204 No Content for favicon requests."""
    from fastapi.responses import Response
    return Response(status_code=204)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
