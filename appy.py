from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

rf_model = joblib.load("src/early_random_forest.pkl")
early_features = joblib.load("src/early_features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    admission_grade = float(request.form["admission_grade"])
    approved_units = float(request.form["approved_units"])
    first_sem_grade = float(request.form["first_sem_grade"])

    return f"""
    Age: {age}<br>
    Admission grade: {admission_grade}<br>
    Approved units: {approved_units}<br>
    First semester grade: {first_sem_grade}
    """

if __name__ == "__main__":
    app.run(debug=True)