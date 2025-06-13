from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

model_path = os.path.join(os.getcwd(), './model/model.pkl')
model = joblib.load(model_path)

FEATURES = [
    'gpa_year', 'edu_level', 'study_time', 'break_freq', 'prep_method', 'sleep_time',
    'exercise', 'sm_use', 'gender', 'tech_use', 'focus', 'phone_use',
    'study_method', 'best_time', 'diet', 'env'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    input_dict = {
        'gpa_year': float(form['gpa_year']),
        'edu_level': form['edu_level'],
        'study_time': form['study_time'],
        'break_freq': form['break_freq'],
        'prep_method': form['prep_method'],
        'sleep_time': form['sleep_time'],
        'exercise': form['exercise'],
        'sm_use': form['sm_use'],
        'gender': form['gender'],
        'tech_use': form['tech_use'],
        'focus': form['focus'],
        'phone_use': form['phone_use'],
        'study_method': ', '.join(request.form.getlist('study_method')),
        'best_time': form['best_time'],
        'diet': form['diet'],
        'env': form['env'],
    }

    input_df = pd.DataFrame([input_dict])

    gpa_pred = model.predict(input_df)[0]
    gpa_pred = round(float(gpa_pred), 2)

    return render_template('predict.html', result=gpa_pred)

if __name__ == '__main__':
    app.run(debug=True)
