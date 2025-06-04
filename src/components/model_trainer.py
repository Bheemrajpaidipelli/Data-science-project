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
from src.utils import save_object, evaluate_models  # Make sure this is correct

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.abspath(os.path.join("artifacts", "model.pkl"))


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainner(self, train_array, test_array):
        try:
            logging.info("Importing train and test array from transformation")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGB Regressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            logging.info("Evaluating models...")
            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            print(" model_report:", model_report)

            if not model_report:
                raise CustomException("Model report is empty — evaluation failed.", sys)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f" Best model: {best_model_name} with R2 score: {best_model_score}")
            logging.info(f"Best model selected: {best_model_name} with R2: {best_model_score}")

            # Skipping performance threshold temporarily
            # if best_model_score < 0.6:
            #     raise CustomException("No suitable model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f" Model saved to: {self.model_trainer_config.trained_model_file_path}")
            logging.info("Model saved successfully.")

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            print(f" Final R2 Score on Test Set: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
