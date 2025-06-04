import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging  
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(" Entered the data ingestion method/component")
        try:
            data = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info(" Dataset read into DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info(" Train-test split initiated")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(" Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# ===============================
#  MAIN BLOCK FOR PIPELINE RUN
# ===============================
if __name__ == "__main__":
    try:
        logging.info(" Starting full pipeline: ingestion -> transformation -> training")

        # Ingestion
        ingestion_obj = DataIngestion()
        train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
        print(" Data ingestion completed.")

        # Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_data_path, test_data_path)
        print(" Data transformation completed.")

        # Model Training
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainner(train_arr, test_arr)
        print(f" Final R2 Score: {r2}")

    except Exception as e:
        print(" Exception occurred in pipeline:", e)
        raise CustomException(e, sys)
