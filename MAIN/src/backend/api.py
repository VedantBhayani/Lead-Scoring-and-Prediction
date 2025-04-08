import os
import glob
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Lead Scoring API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

# Define models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

# Model for lead data input
class LeadInput(BaseModel):
    company_size: str
    lead_source: str
    industry: str
    country: str
    website_visits: int
    email_opens: int
    form_submissions: int
    time_on_site: int
    created_at: Optional[str] = None
    last_activity_date: Optional[str] = None

# Get most recent prediction file
def get_latest_prediction_file():
    prediction_files = glob.glob(os.path.join(OUTPUT_DIR, "predictions_*.csv"))
    if not prediction_files:
        return None
    return max(prediction_files, key=os.path.getctime)

@app.get("/")
def read_root():
    return {"message": "Lead Scoring API is running"}

@app.get("/api/predictions")
def get_predictions():
    """Get predictions from the most recent prediction file"""
    prediction_file = get_latest_prediction_file()
    
    # If no prediction file exists, return mock data
    if not prediction_file:
        return [
            {
                "lead_score": 78,
                "conversion_probability": 0.78,
                "predicted_conversion": True,
                "category": "Warm",
                "feature_importance": {
                    "website_visits": 0.35,
                    "email_opens": 0.25,
                    "form_submissions": 0.20,
                    "time_on_site": 0.20
                },
                "timestamp": datetime.now().isoformat(),
                "company_size": "Medium",
                "industry": "Technology",
                "lead_source": "Website"
            }
        ]
    
    try:
        # Read prediction file
        df = pd.read_csv(prediction_file)
        
        # Convert DataFrame to list of dictionaries
        predictions = df.to_dict('records')
        
        # Process feature_importance if it's stored as a string
        for pred in predictions:
            if 'feature_importance' in pred and isinstance(pred['feature_importance'], str):
                try:
                    pred['feature_importance'] = json.loads(pred['feature_importance'])
                except:
                    pred['feature_importance'] = {}
        
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading prediction file: {str(e)}")

@app.post("/api/predict")
def predict_lead(lead: LeadInput):
    """Make a prediction for a new lead"""
    # In a real implementation, this would call the lead_scoring model
    # For now, we'll just return mock data
    
    # Convert lead to dictionary
    lead_dict = lead.dict()
    
    # Simulate a prediction
    score = min(100, int((lead_dict['website_visits'] * 2 + 
                         lead_dict['email_opens'] * 3 + 
                         lead_dict['form_submissions'] * 10 + 
                         lead_dict['time_on_site'] * 0.1)))
    
    return {
        "lead_score": score,
        "conversion_probability": score / 100,
        "predicted_conversion": score >= 50,
        "category": "Hot" if score >= 80 else "Warm" if score >= 50 else "Cold",
        "feature_importance": {
            "website_visits": 0.3,
            "email_opens": 0.2,
            "form_submissions": 0.4,
            "time_on_site": 0.1
        },
        "timestamp": datetime.now().isoformat(),
        "company_size": lead_dict["company_size"],
        "industry": lead_dict["industry"],
        "lead_source": lead_dict["lead_source"]
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 