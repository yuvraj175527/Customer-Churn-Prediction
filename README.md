# Customer Churn Prediction Pipeline

A Machine Learning-based Customer Churn Prediction System that predicts whether a customer is likely to leave a subscription-based service. The project uses Python, Scikit-learn, and FastAPI to build, train, evaluate, and deploy a predictive machine learning model through a REST API.

## Project Overview

Customer churn is a major challenge for subscription-based businesses such as telecom companies and streaming platforms.

This project uses historical customer data to identify customers who are likely to discontinue their services. It helps businesses understand customer behaviour and take preventive actions to improve customer retention.

The system predicts customer churn probability and classifies customers into different risk levels.

## Key Features

- Data preprocessing and cleaning.
- Handling missing values and duplicate records.
- One-Hot Encoding for categorical variables.
- Numerical feature scaling.
- Feature engineering with at least three additional features.
- Training multiple machine learning classification models.
- Hyperparameter tuning using cross-validation.
- Model evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC and Precision-Recall.
- Best model selection based on performance.
- REST API deployment using FastAPI.
- Real-time customer churn prediction.
- Interactive API documentation using Swagger UI.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| FastAPI | REST API development |
| Uvicorn | API server |
| Joblib | Model serialization |
| HTML, CSS, JavaScript | Frontend interface |

## Project Structure

```text
Customer-Churn-Prediction/
│
├── api/
│   └── main.py
│
├── dataset/
│   └── churn.csv
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The project uses a customer churn dataset containing customer information, subscription details, service usage, and billing information.

Dataset: Telco Customer Churn Dataset.

The dataset is stored inside the `dataset/` directory as:

`churn.csv`

## Machine Learning Models

The project trains and compares multiple classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier

Hyperparameter tuning is performed using cross-validation to improve model performance.

### Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Precision-Recall Analysis

The best-performing model is saved as `best_model.pkl` inside the `models/` directory.

## Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/yuvraj175527/Customer-Churn-Prediction.git
```

### Step 2: Open the Project Directory

```bash
cd Customer-Churn-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

If dependencies are not available in requirements.txt, install the main packages:

```bash
pip install pandas numpy scikit-learn fastapi uvicorn joblib matplotlib
```

### Step 5: Train the Machine Learning Model

Run the following command from the project root directory:

```bash
python src/train.py
```

This process will:

- Load the customer churn dataset.
- Preprocess the data.
- Perform feature engineering.
- Train and evaluate multiple machine learning models.
- Select the best-performing model.
- Save the trained model as `models/best_model.pkl`.

**Note:** The model file is generated automatically after successful training.

### Step 6: Start the FastAPI Server

Run:

```bash
uvicorn api.main:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

## API Documentation

FastAPI provides interactive API documentation through Swagger UI.

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

You can test the prediction API directly through this documentation.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Home endpoint |
| GET | `/health` | API health check |
| POST | `/predict` | Predict customer churn |

### Customer Churn Prediction

The `/predict` endpoint accepts customer information in JSON format and returns the predicted churn probability, binary prediction, and risk level.

Example response:

```json
{
    "churn_probability": 32.49,
    "prediction": "No",
    "risk_level": "Low"
}
```

**Response Explanation:**

- `churn_probability`: Estimated probability of customer churn.
- `prediction`: Binary churn classification (Yes/No).
- `risk_level`: Customer churn risk category.

*Use the request fields defined by the API's prediction schema when testing `/predict`.*

## Running the Project

To run the complete project:

1. Clone the GitHub repository.
2. Install the required dependencies.
3. Place the dataset inside the dataset directory.
4. Train the machine learning models.
5. Start the FastAPI server.
6. Open the Swagger documentation.
7. Test the `/predict` endpoint using customer information.

## Real-World Applications

This project can be used by:

- Telecom companies.
- Subscription-based businesses.
- Streaming platforms.
- Internet service providers.
- Customer relationship management systems.

Businesses can use churn predictions to identify at-risk customers and develop targeted customer retention strategies.

## Future Enhancements

- Integration with live customer databases.
- Advanced machine learning algorithms.
- Automated customer retention recommendations.
- Real-time monitoring dashboard.
- Cloud-based API deployment.

## Author

**Yuvraj Khadka**

GitHub: https://github.com/yuvraj175527

## License

This project is developed for educational and academic purposes.
