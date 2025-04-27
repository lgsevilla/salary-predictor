from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('salary_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        years = float(request.form['years_of_experience'])
        prediction = model.predict(np.array([[years]]))
        predicted_salary = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"Predicted Salary: ${predicted_salary:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
