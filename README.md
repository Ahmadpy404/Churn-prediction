# ChurnSight - AI Powered Customer Churn Prediction Dashboard

ChurnSight is a modern, fully functional web dashboard built with Flask that predicts whether a customer is likely to churn (leave a service) based on their demographics and account information. The dashboard uses a pre-trained Machine Learning model (Logistic Regression) to provide instant predictions and probabilities.

## 🚀 Features

- **Overview Dashboard**: Beautiful, real-time KPI cards displaying Total Customers, Overall Churn Rate, Average Tenure, and Monthly Charges.
- **Interactive Visualizations**: Dynamic charts built with `Chart.js` showing churn distribution across Contract Types, Genders, and Tenure.
- **Prediction Form**: A comprehensive input form to collect 20 different customer features.
- **Real-Time Prediction**: Instantly returns whether the customer is at "High Risk" or "Low Risk" along with the churn probability percentage.
- **Premium UI/UX**: Designed with a sleek Dark Mode, Glassmorphism aesthetics, responsive layouts, and smooth animations using vanilla CSS.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Pandas, Scikit-Learn, Joblib
- **Frontend**: HTML5, Vanilla CSS, JavaScript, Chart.js
- **Model**: Logistic Regression (trained on telecom customer churn dataset)

## 📁 Project Structure

```text
g:/churn_project/
├── app.py                            # Main Flask backend application
├── churn_model.pkl                   # Pre-trained Logistic Regression model
├── customer_data.csv                 # Dataset for generating dashboard metrics
├── churn_prediction_model.ipynb      # Original Jupyter Notebook for training the model
├── requirements.txt                  # Python dependencies
├── static/
│   └── css/
│       └── style.css                 # Premium Glassmorphic CSS styling
└── templates/
    ├── base.html                     # HTML layout template with Sidebar and Header
    ├── dashboard.html                # Overview dashboard with KPIs and Charts
    └── predict.html                  # Churn Prediction Form
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally on your machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/churn-prediction-dashboard.git
   cd churn-prediction-dashboard
   ```

2. **Set up a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:
   ```bash
   python app.py
   ```
   *The application will automatically open your default web browser and navigate to the dashboard (`http://127.0.0.1:5000`).*

## 💡 How it Works

The prediction endpoint takes the raw form inputs and processes them into a pandas DataFrame. It then generates dummy variables (One-Hot Encoding) for categorical columns (like Internet Service, Contract type) and re-indexes them to exactly match the 30 features that the `churn_model.pkl` expects. Finally, it uses `model.predict_proba()` to evaluate the churn risk.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📝 License

This project is open-source and available under the MIT License.
