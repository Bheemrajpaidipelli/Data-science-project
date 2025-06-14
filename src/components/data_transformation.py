import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        "This function is responsible for data transformation"
        try:
            numerical_features = ['reading score', 'math score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))  # Avoids error with sparse matrix
                ]
            )

            logging.info(f'Numerical columns: {numerical_features}')
            logging.info(f'Categorical columns: {categorical_features}')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'writing score'

            input_feature_train_data = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_data = test_data[target_column_name]

            logging.info('Applying preprocessing object to training and testing datasets')
            input_features_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_features_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_data)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
