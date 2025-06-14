from flask import Flask, request, render_template
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            math_score=int(request.form.get('math_score'))
        )
        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        results = pipeline.predict(df)
        return render_template('home.html', results=round(results[0], 2))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
