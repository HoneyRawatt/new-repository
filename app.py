from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS
from models.random_forest import train_random_forest
from models.decision_tree import train_decision_tree
from models.gradient_boosting import train_gradient_boosting
from models.svr import train_svr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

def load_and_prepare_data():
    data = pd.read_csv('salary_prediction_indiandataset.csv')
    data_encoded = pd.get_dummies(data, columns=['Gender', 'Education Level', 'Job Role', 'Location'], drop_first=True)
    X = data_encoded.drop(columns=['Current Salary'])
    y = data_encoded['Current Salary']
    return X, y

def evaluate_models(X, y):
    models = {
        "random_forest": joblib.load('models/random_forest_model.pkl'),
        "decision_tree": joblib.load('models/decision_tree_model.pkl'),
        "gradient_boosting": joblib.load('models/gradient_boosting_model.pkl'),
        "svr": joblib.load('models/svr_model.pkl')
    }

    best_model = None
    best_r2 = -float('inf')
    
    for model_name, model in models.items():
        scaler = joblib.load(f'models/{model_name}_scaler.pkl')
        columns = joblib.load(f'models/{model_name}_columns.pkl')
        
        X_scaled = scaler.transform(X[columns])
        predictions = model.predict(X_scaled)
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)

        print(f"Model: {model_name}")
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
    
    print(f"Best Model: {best_model} with R2: {best_r2}")
    return best_model

X, y = load_and_prepare_data()
best_model_name = evaluate_models(X, y)

def load_model(model_name):
    model = joblib.load(f'models/{model_name}_model.pkl')
    scaler = joblib.load(f'models/{model_name}_scaler.pkl')
    model_columns = joblib.load(f'models/{model_name}_columns.pkl')
    return model, scaler, model_columns

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        model_name = best_model_name
        model, scaler, model_columns = load_model(model_name)

        input_data = pd.DataFrame({
            'Age': [data['age']],
            'Years of Experience': [data['experience']],
            'Gender_' + data['gender']: [1],
            'Education Level_' + data['education']: [1],
            'Job Role_' + data['job_role']: [1],
            'Location_' + data['location']: [1]
        }).reindex(columns=model_columns, fill_value=0)

        input_data_scaled = scaler.transform(input_data)
        predicted_salary = model.predict(input_data_scaled)
        
        return jsonify({"predicted_salary": round(predicted_salary[0], 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
