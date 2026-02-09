# import os
# import sys
# from dataclasses import dataclass
# # from turtle import save

# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer

# from src.components.data_ingestion import DataIngestion
# from src.exception import CustomException
# from src.logger import logging
# import pickle

# from src.utils import save_object


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join(
#         "artifacts", "preprocessor.pkl")
    
# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         try:
#             numerical_columns = ['writing_score', 'reading_score']
#             categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

#             num_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='median')),
#                     ('scaler', StandardScaler())
#                 ]   

#             )

#             cat_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('one_hot_encoder', OneHotEncoder()),
#                     ('scaler', StandardScaler())
#                 ]

#             )

#             logging.info("Numerical columns standard scaling completed")
#             logging.info("Categorical columns encoding completed")

#             preprocessor = ColumnTransformer(
#                 [
#                     ('num_pipeline', num_pipeline, numerical_columns),
#                     ('cat_pipeline', cat_pipeline, categorical_columns)
#                 ]

#             )

#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e, sys)
        
#     def initiate_data_transformation(self, train_path, test_path):

#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed")

#             logging.info("Obtaining preprocessor object")

#             preprocessor_obj = self.get_data_transformer_object()

#             target_column_name = 'math_score'
#             numerical_columns = ['writing_score', 'reading_score']
            
#             input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             logging.info("Applying preprocessor object on training and testing dataframe")

#             input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info("Saving preprocessor object")


#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessor_obj
#             )

#             os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

#             with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
#                 pickle.dump(preprocessor_obj, f)

#             logging.info("Preprocessor object saved successfully")

#             return ( train_arr, 
#                     test_arr,
#                     self.data_transformation_config.preprocessor_obj_file_path
#                     )
#         except Exception as e:
#             raise CustomException(e, sys)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)/


import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'writing_score',
                'reading_score'
            ]

            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = 'math_score'

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing on train and test data")

            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
