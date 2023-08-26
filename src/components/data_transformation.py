import sys
sys.path.insert(0,"../src")

from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


# class DataTransformations:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         try:
#             numerical_columns = ["writing_score","reading_score"]
#             categorical_columns = [
#                 "gender",
#                 'parental_level_of_education',
#                 "lunch",
#                 "test_preparation_course",
#                 "race_ethnicity"

#             ]

#             num_piplines = Pipeline(
#                 steps=[
#                     ("impute",SimpleImputer(strategy="median")),
#                     ("scaler",StandardScaler(with_mean=False))
#                 ]
#             )
            
#             cat_piplines = Pipeline(
#                 steps=[
#                     ("impute",SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder",OneHotEncoder()),
#                     ("scaler",StandardScaler())
#                 ]
#             )

#             logging.info(f"Numerical columns: {categorical_columns}")
            
#             logging.info(f"Categorical Columns: {numerical_columns}")

#             preprocesser=ColumnTransformer(
#                 [
#                     ("num_pipline",num_piplines,numerical_columns),
#                     ("cat_pipelines",cat_piplines,categorical_columns )
#                 ]
#             )

#             return preprocesser

#         except Exception as e:
#             raise CustomException(e,sys)
            

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformations:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "race_ethnicity"  # Make sure the column name matches your data
            ]

            num_pipelines = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipelines = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipelines, numerical_columns),
                    ("cat_pipelines", cat_pipelines, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Optaining preprocessing object")

            preprocessering_obj=self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on trainning dataframe and testing dataframe."

            )
            input_feature_train_arr=preprocessering_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessering_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr ,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr ,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocesser_obj_file_path, 
                obj = preprocessering_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path,
                )  
        except Exception as e:
            raise CustomException(e,sys)