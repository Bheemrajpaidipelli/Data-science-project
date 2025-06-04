import os
import sys
import dill
import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

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


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """
    Trains and tunes multiple models using GridSearchCV.
    Returns:
        - report: dict with model names and test R² scores
        - trained_models: dict with model names and fitted model objects
    """
    try:
        report = {}
        trained_models = {}

        for name, model in models.items():
            logging.info(f"Training and tuning model: {name}")

            # Get hyperparameter grid for this model
            model_param_grid = params.get(name, {})

            # Perform grid search
            gs = GridSearchCV(model, model_param_grid, cv=3, n_jobs=-1, verbose=1)
            gs.fit(x_train, y_train)

            # Use best estimator from grid search
            best_model = gs.best_estimator_

            # Predict using best model
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Compute R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")

            report[name] = test_model_score
            trained_models[name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
