from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('../notebooks/final_logistic_regression_model.pkl')

@app.route('/')
def home():
    # Render index.html as the home page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Convert prediction to response
    output = round(prediction[0], 2)
    
    # Return the result
    return render_template('index.html', prediction_text=f'Customer Churn Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)