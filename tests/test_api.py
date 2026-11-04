
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# Test health endpoint
def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test customer prediction
def test_prediction():

    data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "prediction" in result
    assert "risk_level" in result

    assert 0 <= result["churn_probability"] <= 100

    assert result["prediction"] in ["Yes", "No"]

    assert result["risk_level"] in [
        "Low", "Medium", "High"
    ]


# Test invalid input
def test_invalid_customer():

    data = {
        "gender": "Unknown"
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 422
