import os
import sys
import dill
import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves a Python object to disk using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f'Object has been serialized and saved to {file_path}.')

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    Trains multiple models and returns a report dictionary with R² scores.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)  

            logging.info(f"{name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")
            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
