from flask import Flask, render_template, request
import joblib
import pandas as pd

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

    previous_grade = float(request.form["previous_grade"])
    debtor = int(request.form["debtor"])
    tuition_up_to_date = int(request.form["tuition_up_to_date"])
    scholarship = int(request.form["scholarship"])
    international = int(request.form["international"])

    first_sem_enrolled = float(request.form["first_sem_enrolled"])
    first_sem_evaluations = float(request.form["first_sem_evaluations"])
    first_sem_credited = float(request.form["first_sem_credited"])
    first_sem_without_evaluations = float(request.form["first_sem_without_evaluations"])

    application_mode = int(request.form["application_mode"])
    application_order = int(request.form["application_order"])
    course_name = request.form["course_name"]
    attendance = int(request.form["attendance"])
    previous_qualification = int(request.form["previous_qualification"])
    displaced = int(request.form["displaced"])
    special_needs = int(request.form["special_needs"])
    gender = int(request.form["gender"])

    course_codes = {
            "Biofuel Production Technologies": 33,
            "Animation and Multimedia Design": 171,
            "Social Service (evening attendance)": 8014,
            "Agronomy": 9003,
            "Communication Design": 9070,
            "Veterinary Nursing": 9085,
            "Informatics Engineering": 9119,
            "Equinculture": 9130,
            "Management": 9147,
            "Social Service": 9238,
            "Tourism": 9254,
            "Nursing": 9500,
            "Oral Hygiene": 9556,
            "Advertising and Marketing Management": 9670,
            "Journalism and Communication": 9773,
            "Basic Education": 9853,
            "Management (evening attendance)": 9991 
        }
    
    course = course_codes[course_name]

    student_data = {
       "Application mode": application_mode,
       "Application order": application_order,
       "Course": course,
       "Daytime/evening attendance\t": attendance,
       "Previous qualification": previous_qualification,
       "Previous qualification (grade)": previous_grade,
       "Admission grade": admission_grade,
       "Displaced": displaced,
       "Educational special needs": special_needs,
       "Debtor": debtor,
       "Tuition fees up to date": tuition_up_to_date,
       "Gender": gender,
       "Scholarship holder": scholarship,
       "Age at enrollment": age,
       "International": international,
       "Curricular units 1st sem (credited)": first_sem_credited,
       "Curricular units 1st sem (enrolled)": first_sem_enrolled,
       "Curricular units 1st sem (evaluations)": first_sem_evaluations,
       "Curricular units 1st sem (approved)": approved_units,
       "Curricular units 1st sem (grade)": first_sem_grade,
       "Curricular units 1st sem (without evaluations)": first_sem_without_evaluations
    }


    student_df = pd.DataFrame([student_data])
    student_df = student_df[early_features]

    dropout_probability = rf_model.predict_proba(student_df)[0][1]
    dropout_percentage = dropout_probability * 100

    if dropout_probability < 0.30:
        risk_level = "LOW"
    elif dropout_probability < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return render_template(
        "result.html",
        dropout_percentage=round(dropout_percentage, 1),
        risk_level=risk_level
    )


if __name__ == "__main__":
    app.run(debug=True)