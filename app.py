# app.py
import os
import time
import datetime
import schedule
from threading import Thread
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import torch
import shutil
import subprocess
import uuid
import uvicorn
import requests
from pydantic import BaseModel, validator
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app, CollectorRegistry
from model import FashionCNN
from data_preprocessing import preprocess_user_image, save_user_image
from fastapi import Body
#from monitoring import ModelMonitor
from enum import Enum
from typing import Optional
from evidently.report import Report
from evidently.metrics import DataDriftTable

class ClassLabels(str, Enum):
    t_shirt = "T-shirt/top"
    trouser = "Trouser"
    pullover = "Pullover"
    dress = "Dress"
    coat = "Coat"
    sandal = "Sandal"
    shirt = "Shirt"
    sneaker = "Sneaker"
    bag = "Bag"
    ankle_boot = "Ankle boot"


registry = CollectorRegistry()
#Drift Detection Functions
data_drift_score = Gauge('data_drift_score', 'Overall data drift score', registry=registry)
feature_drift_count = Gauge('feature_drift_count', 'Number of drifting features', registry=registry)


    
def load_reference_data(path):
    """Load reference Fashion-MNIST data"""
    from torchvision.datasets import FashionMNIST
    dataset = FashionMNIST(root=path, train=True, download=True)
    return pd.DataFrame(dataset.data.numpy().reshape(-1, 784))

ref_data=load_reference_data("data/FashionMNIST")

def check_drift_evidently(current_data):
    """Detect drift using Evidently AI"""
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=ref_data, current_data=current_data)
    return report.as_dict()['metrics'][0]['result']

DRIFT_BATCH = []
def update_metrics(current_batch):
    global DRIFT_BATCH
    
    # Accumulate batches (e.g., 100 samples)
    DRIFT_BATCH.append(current_batch.numpy().reshape(784))
    
    # Check if ready for drift calculation
    if len(DRIFT_BATCH) >= 10:
        current_data = pd.DataFrame(np.array(DRIFT_BATCH))
        
        # Evidently detection
        evidently_result = check_drift_evidently(current_data)
        data_drift_score.set(evidently_result['dataset_drift_score'])
        feature_drift_count.set(
            sum([v['drift_detected'] for v in evidently_result['drift_by_columns'].values()])
        )
        
        # Reset batch
        DRIFT_BATCH.clear()

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Classifier API",
    description="API for classifying fashion items in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
   CORSMiddleware,
    allow_origins= ["*"],  # In production, specify exact origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define metrics (previously in monitoring.py)
prediction_counter = Counter('fashion_classifier_predictions_total', 'Number of prediction requests received',['result'],registry=registry)
prediction_latency = Histogram('fashion_classifier_prediction_latency_seconds', 'Time taken to make predictions', buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5], registry=registry)
model_accuracy = Gauge('fashion_classifier_model_accuracy_percent', 'Current estimated model accuracy percentage', registry=registry)

# Mount the metrics endpoint to FastAPI
# Update model accuracy from metadata
def update_model_accuracy():
    """Load and update model accuracy from metadata"""
    try:
        import json
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            accuracy = metadata.get("accuracy", 0)
            model_accuracy.set(accuracy)
    except (FileNotFoundError, json.JSONDecodeError):
        model_accuracy.set(0)
        #accuracy = 0

# Load the trained model
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FashionCNN().to(device)
    
    try:
        model.load_state_dict(torch.load("models/fashion_model.pth", map_location=device))
        model.eval()
        # Update accuracy metric when model is loaded
        update_model_accuracy()
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Define class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# Define response models
class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    prediction_time_ms: float

class FeedbackRequest(BaseModel):
    image_id: str
    is_correct: bool
    predicted_class: str
    actual_class: Optional[ClassLabels]=None

    @validator('predicted_class', 'actual_class')
    def validate_class_names(cls, v):
        valid_classes = [e.value for e in ClassLabels]
        if v and v not in valid_classes:
            raise ValueError(f"Invalid class name. Valid options :{valid_classes}")
        return v

def trigger_retrain_job():
    """Helper function to call retrain endpoint"""
    api_key = os.getenv("RETRAIN_API_KEY", "your-secret-key")
    try:
        response = requests.post("http://localhost:5001/retrain", data={"api_key": api_key})
        if response.status_code != 200:
            print(f"Retrain trigger failed: {response.text}")
    except Exception as e:
        print(f"Retrain trigger error: {str(e)}")

def run_scheduled_checks():
    schedule.every().hour.at(":00").do(trigger_retrain_job)
    schedule.every(6).hours.do(check_drift_evidently)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler():
    Thread(target=run_scheduled_checks, daemon=True).start()

# API endpoints
@app.on_event("startup")
async def startup_event():
    start_scheduler()

@app.get("/")
async def root():
    return {"message": "Fashion Classification API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Endpoint for classifying fashion items in images"""
    current_model_version = os.path.getmtime("models/fashion_model.pth")
    if not hasattr(predict_image, "last_model_version") or predict_image.last_model_version < current_model_version:
        reload_model(os.getenv("RELOAD_KEY"))
        predict_image.last_model_version = current_model_version
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Only JPEG and PNG images are supported"}
            )

        # Create temp directory with proper permissions
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True, mode=0o755)

        # Generate unique filename to prevent collisions
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = f"{timestamp}_{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = os.path.join(temp_dir, unique_id)
        file_location = f"temp/{unique_id}"

        # Save uploaded file with proper error handling
        with open(temp_file_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify model is loaded
        if model is None:
            prediction_counter.labels(result="error").inc()
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded. Please train the model first."}
            )

        # Process image and make prediction
        start_time = time.time()
        processed_tensor = preprocess_user_image(temp_file_path)
        
        # Move tensor to same device as model
        device = next(model.parameters()).device
        input_tensor = processed_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = class_labels[predicted_idx.item()]
            confidence = torch.exp(output[0][predicted_idx]).item() * 100

        # Calculate processing time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        # Record metrics
        prediction_counter.labels(result="success").inc()
        prediction_latency.observe(end_time - start_time)

        
        #update_metrics(processed_tensor)

        return {
            "filename": unique_id,
            "predicted_class": predicted_label,
            "confidence": confidence,
            "prediction_time_ms": processing_time_ms
        }
        
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        prediction_counter.labels(result="error").inc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest = Body(...)):
    """Process user feedback to improve the model"""
    try:
        # Validate input
        if not feedback.is_correct and not feedback.actual_class:
            return JSONResponse(
                status_code=400,
                content={"error": "Actual class required for incorrect predictions"}
            )

        # Construct temp file path using the unique image_id
        temp_file_path = f"temp/{feedback.image_id}"
        
        if not os.path.exists(temp_file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Image not found. It may have already been processed."}
            )

        # Determine class name and destination directory
        if feedback.is_correct:
            class_name = feedback.predicted_class.replace("/", "-")  # Sanitize filename
            destination_dir = "data/user-images/correct"
        else:
            class_name = feedback.actual_class.replace("/", "-")  # Sanitize filename
            destination_dir = "data/user-images/incorrect"

        # Create destination path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{timestamp}_{feedback.image_id}"
        destination_path = os.path.join(destination_dir, filename)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Move the file instead of copying to avoid duplicates
        shutil.move(temp_file_path, destination_path)

        # Add to DVC tracking
        subprocess.run(["dvc", "add", "/app/data/user-images"])

        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "saved_path": destination_path
        }

    except Exception as e:
        # Cleanup if error occurred mid-process
        if os.path.exists(temp_file_path):
            shutil.move(temp_file_path, f"temp/error_{feedback.image_id}")
            
        return JSONResponse(
            status_code=500,
            content={"error": f"Feedback processing failed: {str(e)}"}
        )

@app.post("/reload-model")
async def reload_model(api_key: str = Form(...)):
    """Hot-reload the production model"""
    if api_key != os.getenv("RELOAD_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    global model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FashionCNN().to(device)
        model.load_state_dict(torch.load("models/fashion_model.pth"))
        update_model_accuracy()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/health")
async def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "healthy", "message": "API is running correctly"}
#Add retrain method
@app.post("/retrain")
async def trigger_retraining(api_key: str = Form(...)):
    """Endpoint to trigger model retraining (secured with API key)"""
    # Verify API key (simple implementation - use proper auth in production)
    if api_key != os.environ.get("RETRAIN_API_KEY", "your-secret-key"):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    
    # Run retraining in background
    import subprocess
    try:
        subprocess.Popen(["python", "retrain.py"])
        return {"status": "success", "message": "Retraining started"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start retraining: {str(e)}"}
        )
#Method for pushing & pulling images to and from DVC
@app.post("/dvc/push")
async def push_dvc_data(api_key: str = Form(...)):
    """Push DVC data to remote storage"""
    if api_key != os.environ.get("DVC_API_KEY", "your-secret-key"):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    
    from dvc_helper import push_data
    success = push_data()
    
    if success:
        return {"status": "success", "message": "Data pushed to remote storage"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to push data"}
        )
#
@app.post("/dvc/pull")
async def pull_dvc_data(api_key: str = Form(...)):
    """Pull DVC data from remote storage"""
    if api_key != os.environ.get("DVC_API_KEY", "your-secret-key"):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    
    from dvc_helper import pull_data
    success = pull_data()
    
    if success:
        return {"status": "success", "message": "Data pulled from remote storage"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to pull data"}
        )
metrics_app = make_asgi_app(registry=registry)
app.mount("/metrics", metrics_app)
# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
