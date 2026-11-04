<<<<<<< HEAD
# Customer Churn Prediction Pipeline

End-to-end supervised machine learning project for predicting customer churn.

## Project structure
- `dataset/`: place the churn CSV dataset here.
- `models/`: trained model artifacts are saved here.
- `src/`: data preprocessing, feature engineering, training and evaluation.
- `api/`: FastAPI application.
- `notebooks/`: optional exploratory analysis.
- `tests/`: project tests.

## Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Run
Training and API commands will be added as the pipeline is implemented.
=======
# Customer Churn Prediction System

## Project Overview

The **Customer Churn Prediction System** is a machine learning-based application designed to predict whether a customer is likely to leave a telecom service provider.

The project uses real-world telecom customer data to analyze customer behavior, subscription details, service usage, and billing information. It applies multiple machine learning algorithms to identify potential churners and helps businesses understand customer retention risks.

The trained machine learning model is deployed using **FastAPI**, allowing users to submit customer information through an interactive web dashboard and receive churn predictions.

## Key Features

- **Data Preprocessing:** Handles missing values, categorical variables, and numerical data.
- **Feature Engineering:** Creates additional features such as Tenure Group, Average Monthly Spend, and Total Services.
- **Machine Learning:** Implements Logistic Regression, Random Forest, and Gradient Boosting.
- **Hyperparameter Tuning:** Uses GridSearchCV with cross-validation to optimize model performance.
- **Model Evaluation:** Evaluates performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
- **Business Impact Analysis:** Estimates the potential costs associated with missed churners and false alarms using configurable assumptions.
- **REST API Deployment:** Deploys the trained model using FastAPI.
- **Input Validation:** Uses Pydantic to validate customer information.
- **Interactive Dashboard:** Provides a web interface for entering customer details and viewing predictions.
- **Automated Testing:** Uses Pytest to test API functionality.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Storage | Joblib |
| Backend | FastAPI |
| API Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest, HTTPX |
| Development Tools | VS Code, Jupyter Notebook |

## Machine Learning Algorithms

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

The models are compared using cross-validation and ROC-AUC scoring. The best-performing model is saved and used for customer churn predictions.

## Model Performance

Previously recorded evaluation results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 81% |
| ROC-AUC | 84.49% |
| Churn Precision | 68% |
| Churn Recall | 51% |

*Note: These are results from the previously completed training run. Results may change when the model is retrained.*

## Feature Engineering

The system creates three additional customer features:

- **TenureGroup:** Categorizes customers based on subscription duration.
- **AverageMonthlySpend:** Estimates average monthly spending using total charges and tenure.
- **TotalServices:** Counts the number of selected customer services.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Opens the web dashboard |
| GET | `/health` | Checks API status |
| POST | `/predict` | Predicts customer churn |
| GET | `/docs` | Opens Swagger API documentation |

## Prediction Output

The API returns:

- Churn Probability
- Prediction (Yes/No)
- Risk Level (Low/Medium/High)

Example response:

```json
{
  "churn_probability": 32.49,
  "prediction": "No",
  "risk_level": "Low"
}
```

## Project Objective

The main objective is to develop an end-to-end machine learning solution that helps telecom businesses identify customers at risk of leaving, understand churn patterns, and make more informed customer retention decisions.

## Future Enhancements

- Improve churn recall through threshold optimization and model tuning.
- Add customer prediction history.
- Develop advanced analytics and visualization features.
- Provide personalized customer retention recommendations.
- Deploy the application to a cloud platform.

## Conclusion

This project demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, model selection, hyperparameter tuning, model evaluation, business impact analysis, REST API development, and frontend integration.
>>>>>>> 33670ab33b8a0f27874ee62be77ceb1931e4ef00
