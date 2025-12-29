import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# Function to load the machine learning model
def load_model(model_path):
    """Load a machine learning model from the specified file path."""
    model = joblib.load(model_path)
    return model
model = load_model('D:\Amit\Projects\Heart-failure\model\log_.joblib')

@app.route('/', methods=['GET'])
def show():
    return render_template('predict.html')

#Endpoint for health check
@app.route('/health', methods=['POST'])
def health_check():
    Age=request.form.get('Age')
    Sex=request.form.get('Sex')
    ChestPainType=request.form.get('ChestPainType')
    RestingBP=request.form.get('RestingBP')
    Cholesterol=request.form.get('Cholesterol')
    FastingBS=request.form.get('FastingBS')
    RestingECG=request.form.get('RestingECG')
    MaxHR=request.form.get('MaxHR')
    ExerciseAngina=request.form.get('ExerciseAngina')
    Oldpeak=request.form.get('Oldpeak')
    ST_Slope=request.form.get('ST_Slope')


    example_input = pd.DataFrame({
    "Age": [int(Age)],
    "Sex": [Sex],
    "ChestPainType": [ChestPainType],
    "RestingBP": [int(RestingBP)],
    "Cholesterol": [float(Cholesterol)],
    "FastingBS": [int(FastingBS)],
    "RestingECG": [RestingECG],
    "MaxHR": [int(MaxHR)],
    "ExerciseAngina": [ExerciseAngina],
    "Oldpeak": [float(Oldpeak)],
    "ST_Slope": [ST_Slope]
})
    prediction = model.predict(example_input)[0]

    if prediction == 1:
        result = "⚠️ The patient is likely to have heart disease."
        risk = "high"
    else:
        result = "✅ The patient is unlikely to have heart disease."
        risk = "low"

    return render_template(
        'result.html',
        prediction=result,
        risk=risk
    )




app.run()


