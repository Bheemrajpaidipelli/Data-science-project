# End to End Data Science Project 

### problem statemet "
Developed a complete end-to-end Data Science project, encompassing data collection, preprocessing, model training, evaluation, and deployment using Flask (or FastAPI) to build a user-friendly web interface for real-time predictions."

# Table of Contents
1) Student Exam Performance Indicator

2) Problem Statement

3) Project Overview

4) Dataset Information

5) Project Structure

6) Features

7) Model Training and Evaluation

8) Tech Stack

9) Web Interface

10) How to Run the Project

11) Sample Input

12) Sample Output

13 Contributers

# Project Title
Project Title: **Student Exam Performance Indicator**

This is an end-to-end data science project aimed at predicting students' writing scores based on various demographic and academic inputs using a machine learning model. The project involves data preprocessing, model training, evaluation, and deployment using a Flask web application.


# Project Overview
The goal of this project is to:

Predict a student's exam performance (writing score) based on features such as gender, ethnicity, parental education level, lunch type, test preparation course, and scores in math and reading.

Build a user-friendly web interface using Flask to allow users to input data and receive predictions in real time.

# Dataset Information
The dataset includes various features about students:

Gender
Race/Ethnicity
Parental Level of Education
Lunch Type
Test Preparation Course
Reading and Math Scores
Writing  Score (Target Variable)

# Project Structure
'''Student-Performance-Predictor/
│
├── artifacts/# Saved model and preprocessor files
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── templates/     # HTML templates for Flask app
│   ├── index.html
│   └── home.html
│
├── src/                   Source code
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app.py        # Flask application entry point
├── requirements.txt      # Python dependencies
└── README.md          # Project documentation'''


# Features
- Data ingestion and preprocessing

- Model training and evaluation using Linear Regression

- Custom data pipeline for making predictions

- Real-time prediction with a Flask-based frontend

- Error handling and logging

# Model Training and Evaluation
The model pipeline includes:

Data preprocessing with encoding of categorical features
Model training Hyperparameter optimization using GridSearchCVPerformance evaluation using R² score metricsModel deployment using Flask


# Tech Stack
Languages: Python

Libraries: Pandas, Scikit-learn, NumPy, Matplotlib
 seaborn.

Web Framework: Flask

Model Persistence: Pickle

Deployment: Localhost

# Web Interface
### Home Page
A welcome page with a link to the prediction form.

### Prediction Page
A form to input student features:

Gender

Ethnicity

Parental level of education

Lunch type

Test preparation course

Reading score

Math score

Displays the predicted writing score upon submission.

# How to Run the Project
1) Clone the repository

git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor

2) Install dependencies

 pip install -r requirements.txt

3) Run the Flask app

python app.py

4) Visit in browser

http://127.0.0.1:5000/

# Sample Input

| Feature                     | Value      |
| --------------------------- | ---------- |
| Gender                      | Female     |
| Race/Ethnicity              | Group B    |
| Parental Level of Education | Bachelor's |
| Lunch                       | Standard   |
| Test Preparation Course     | Completed  |
| Reading Score               | 85         |
| Math Score                  | 78         |

# Sample Output

 Predicted Writing Score: 82.6

# Contributers
- Bheemrajpaidipelli