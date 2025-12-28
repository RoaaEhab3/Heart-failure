import joblib
import pandas as pd

def load_model(model_path):
    """Load a machine learning model from the specified file path."""
    model = joblib.load(model_path)
    return model
model = load_model('D:\Amit\Projects\Heart-failure\model\log_.joblib')

example_input = pd.DataFrame({
    "Age": [90],
    "Sex": [1],
    "ChestPainType": ["ASY"],
    "RestingBP": [130],
    "Cholesterol": [250],
    "FastingBS": [1],
    "RestingECG": ["ST"],
    "MaxHR": [82],
    "ExerciseAngina": ["N"],
    "Oldpeak": [0],
    "ST_Slope": ["Flat"]
})
prediction = model.predict(example_input)[0]
if prediction == 1:
    prediction = "The patient is likely to have a heart disease."
else:
    prediction = "The patient is likely not to have a heart disease."
print(prediction)