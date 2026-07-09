import os
import pickle
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('churn_model.pkl')
    # Use the known features
    EXPECTED_FEATURES = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
        'MultipleLines_No phone service', 'MultipleLines_Yes', 
        'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No internet service', 'TechSupport_Yes', 
        'StreamingTV_No internet service', 'StreamingTV_Yes', 
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
        'PaymentMethod_Mailed check'
    ]
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    EXPECTED_FEATURES = []

# Load data for dashboard metrics
try:
    df = pd.read_csv('customer_data.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame()

@app.route('/')
def dashboard():
    """Render the main dashboard with charts and KPIs."""
    if df.empty:
        return "Dataset not found. Cannot render dashboard metrics."
    
    # Calculate KPIs
    total_customers = len(df)
    churn_rate = round((df['Churn'] == 'Yes').mean() * 100, 1)
    avg_tenure = round(df['tenure'].mean(), 1)
    avg_monthly = round(df['MonthlyCharges'].mean(), 2)

    return render_template('dashboard.html', 
                           kpi={'total': total_customers, 
                                'churn_rate': churn_rate, 
                                'avg_tenure': avg_tenure, 
                                'avg_monthly': avg_monthly})

@app.route('/predict_page')
def predict_page():
    """Render the prediction form."""
    return render_template('predict.html')

@app.route('/api/chart_data')
def chart_data():
    """API endpoint to serve data for charts."""
    if df.empty:
        return jsonify({})
    
    # Churn by Contract
    contract_churn = df.groupby('Contract')['Churn'].value_counts().unstack().fillna(0)
    contract_data = {
        'labels': contract_churn.index.tolist(),
        'yes': contract_churn['Yes'].tolist(),
        'no': contract_churn['No'].tolist()
    }
    
    # Churn by Gender
    gender_churn = df.groupby('gender')['Churn'].value_counts().unstack().fillna(0)
    gender_data = {
        'labels': gender_churn.index.tolist(),
        'yes': gender_churn['Yes'].tolist(),
        'no': gender_churn['No'].tolist()
    }
    
    # Tenure Distribution
    # group by tenure bins
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-1 yr', '1-2 yrs', '2-3 yrs', '3-4 yrs', '4-5 yrs', '5-6 yrs']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    tenure_churn = df.groupby('tenure_group')['Churn'].value_counts().unstack().fillna(0)
    tenure_data = {
        'labels': labels,
        'yes': tenure_churn['Yes'].tolist(),
        'no': tenure_churn['No'].tolist()
    }

    return jsonify({
        'contract': contract_data,
        'gender': gender_data,
        'tenure': tenure_data
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request from the form."""
    if not model:
        return jsonify({'error': 'Model not loaded.'})
    
    data = request.form.to_dict()
    
    # Convert form inputs to the format expected by the model
    try:
        input_dict = {
            'gender': 1 if data.get('gender') == 'Male' else 0,
            'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
            'Partner': 1 if data.get('Partner') == 'Yes' else 0,
            'Dependents': 1 if data.get('Dependents') == 'Yes' else 0,
            'tenure': float(data.get('tenure', 0)),
            'PhoneService': 1 if data.get('PhoneService') == 'Yes' else 0,
            'PaperlessBilling': 1 if data.get('PaperlessBilling') == 'Yes' else 0,
            'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
            'TotalCharges': float(data.get('TotalCharges', 0)),
            'MultipleLines': data.get('MultipleLines'),
            'InternetService': data.get('InternetService'),
            'OnlineSecurity': data.get('OnlineSecurity'),
            'OnlineBackup': data.get('OnlineBackup'),
            'DeviceProtection': data.get('DeviceProtection'),
            'TechSupport': data.get('TechSupport'),
            'StreamingTV': data.get('StreamingTV'),
            'StreamingMovies': data.get('StreamingMovies'),
            'Contract': data.get('Contract'),
            'PaymentMethod': data.get('PaymentMethod')
        }
        
        # Create a single-row DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Separate numeric and categorical for dummies processing
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        # Generate dummy variables
        input_dummies = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Reindex to ensure it matches the model's expected features exactly
        # Fill missing dummy columns with 0
        final_input = input_dummies.reindex(columns=EXPECTED_FEATURES, fill_value=0)
        
        # Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1] # Probability of Churn=Yes (Class 1)
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(probability * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    # Ensure it only opens the browser once (not in the reloader process)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        import webbrowser
        from threading import Timer
        Timer(1.25, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
        
    app.run(debug=True, port=5000)
