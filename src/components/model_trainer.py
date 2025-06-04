import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.abspath(os.path.join("artifacts", "model.pkl"))


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainner(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data arrays.")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGB Regressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            params = {
                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "XGB Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            logging.info("Evaluating models using GridSearchCV...")
            model_report, trained_models = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            if not model_report:
                raise CustomException("Model report is empty — evaluation failed.", sys)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = trained_models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with R² score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Final R² score on test set: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
