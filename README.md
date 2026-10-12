# EduPredict_AI
Machine learning project for early prediction of student dropout risk using academic and demographic data.

The project combines data analysis, machine learning, and a Flask web application that allows users to enter student information and receive an estimated dropout risk.

## Project Objective
The main goal is to explore whether students at risk of dropping out can be identified early enough to allow possible academic intervention. 

Two machine Learning approaches were compared:

- Early Model - uses enrollment and first-semester information.
- Full Model - includes additional second-semester information.

## Dataset
The project uses the UCI dataset:

**Predict Students' Dropout and Academic Success**

The dataset contains information about 4,424 students from a Portuguese higher-education institution.
It includes:
- demographic information
- admission information
- previous qualifications
- financial information
- first-semester academic performance
- second-semester academic performance
- final student outcome

The original target contains three classes;
- Graduate
- Dropout
- Enrolled

For this project, the target was converted into a binary classification problem:
- '1' = Dropout
- '2' = Not Dropout

## Exploratory Data Analysis

Initial analysis showed that students who dropped out generally had:
- fewer approved curricular units
- lower first-semester grades
- slightly lower admission grades
- higher average age at enrollment

First-semester academic performance appeared to be one of the strongest sources of predictive information.


## Machine Learning Models

### Logistic Regression
A Logistic Regression model was used as a baseline.

Early Model performance:

- Accuracy: approximately 85.2%
- Dropout Recall: approximately 66%

### Early Random Forest

The Early Random Forest uses enrollment and first-semester information.

Performance:

- Accuracy: approximately 86.2%
- Dropout precision: approximately 85.2%
- Dropout Recall: approximately 69.0%
- Dropout f1-score: approximately 76.3%

### Full Random Forest

The Full Random Forest also includes second-semester information.

Performance:

- Accuracy: approximately 88.2%
- Dropout precision: approximately 87.2%
- Dropout Recall: approximately 74.3%
- Dropout f1-score: approximately 80.2%

## Important Features

Some of the most important features in the Early Random Forest model were:

- first-semester approved curricular units
- first-semester average grade
- tuition fee status
- admission grade
- age at enrollment
- first-semester evaluations
- previous qualification grade

Feature importance represents predictive usefulness and does not imply causation.

## Web Application

The project includes a Flask web application.

Users can enter student information such as:

- age
- addmission grade
- application mode
- previous qualification 
- financial information
- first-semester academic performance

The application then returns:

- estimated dropout risk
- risk level: LOW, MEDIUM, or HIGH

## Technologies

- Python
- pandas
- scikit-learn
- Random Forest
- Logistic Regression
- Flask
- HTML
- CSS
- Jupyter Notebook
- joblib