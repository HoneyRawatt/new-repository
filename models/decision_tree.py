import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split

def train_decision_tree(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Decision Tree Regressor
    model = DecisionTreeRegressor(max_depth=5, random_state=42)  # Limiting depth to avoid overfitting
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and calculate Mean Absolute Error
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    # Save the trained model, scaler, and feature columns
    joblib.dump(model, 'models/decision_tree_model.pkl')
    joblib.dump(scaler, 'models/decision_tree_scaler.pkl')
    joblib.dump(X.columns, 'models/decision_tree_columns.pkl')

# Example usage:
# Load your dataset and prepare X (features) and y (target variable)
data = pd.read_csv('salary_prediction_indiandataset.csv')
data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop(columns=['Current Salary'])
y = data_encoded['Current Salary']

# Train and save the Decision Tree model
train_decision_tree(X, y)
