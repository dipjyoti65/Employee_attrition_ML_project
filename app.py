from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the Machine Learning model (change the path accordingly)
with open('randomforest_modelY.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
# values from the home.html file is assigned

def predict():
    if request.method == 'POST':

        age = request.form['Age']
        environment_satisfaction = request.form['EnvironmentSatisfaction']
        job_involvement = request.form['JobInvolvement']
        job_level = request.form['JobLevel']
        overtime = request.form['OverTime']
        # relationship_satisfaction = request.form['RelationshipSatisfaction']
        shift = request.form['Shift']
        total_working_years = request.form['TotalWorkingYears']
        training_times_last_year = request.form['TrainingTimesLastYear']
        years_at_company = request.form['YearsAtCompany']
        years_in_current_role = request.form['YearsInCurrentRole']
        years_with_curr_manager = request.form['YearsWithCurrManager']
        cat_therapist = request.form['cat_Therapist']
        cat_married = request.form['cat_Married']
        cat_single = request.form['cat_Single']
        salary = request.form['cat_high']

        # Create a DataFrame from the input data to send to the model
        input_data = pd.DataFrame({
            'Age': [age],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'OverTime': [overtime],
            'Shift': [shift],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsWithCurrManager': [years_with_curr_manager],
            'cat_Therapist': [cat_therapist],
            'cat_Married': [cat_married],
            'cat_Single': [cat_single],
            'cat_high': [salary]
        })

        try:
            prediction_str = prediction(input_data)
            probability_str = probability(input_data)
            return render_template('result.html', prediction=prediction_str, probability=probability_str)
        except Exception as e:
            return jsonify({'error': str(e)})


def prediction(input_data):
    prediction_value = model.predict(input_data)
    temp = prediction_value[0]
    return str(temp)


def probability(input_data):
    probability_value = model.predict_proba(input_data)
    temp = probability_value[0][1] * 100
    return str(temp)


if __name__ == '__main__':
    app.run(debug=True)
