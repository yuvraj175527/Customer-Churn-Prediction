
from pathlib import Path
import logging
import joblib
import pandas as pd

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.feature_engineering import create_features


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using Machine Learning",
    version="1.0.0"
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"


# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run: python src/train.py"
    )

model = joblib.load(MODEL_PATH)

logger.info("Machine learning model loaded successfully")


# Customer input validation
class Customer(BaseModel):

    gender: Literal["Male", "Female"]

    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]

    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]

    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Customer input fields
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]


# Home endpoint
@app.get("/")
def home():

    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: Customer):

    try:

        # Convert validated customer data
        customer_data = customer.model_dump()

        data = pd.DataFrame(
            [customer_data],
            columns=FEATURES
        )

        # Apply shared feature engineering
        data = create_features(data)

        # Generate prediction
        probability = float(
            model.predict_proba(data)[0][1]
        )

        prediction = int(probability >= 0.5)

        if probability >= 0.65:
            risk = "High"

        elif probability >= 0.40:
            risk = "Medium"

        else:
            risk = "Low"

        logger.info("Customer prediction completed")

        return {
            "churn_probability": round(
                probability * 100, 2
            ),
            "prediction": (
                "Yes" if prediction else "No"
            ),
            "risk_level": risk
        }

    except Exception:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process customer prediction"
        )


# Serve frontend
if FRONTEND_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static"
    )
