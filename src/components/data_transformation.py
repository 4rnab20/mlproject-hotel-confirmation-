import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('output',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
    # function for data transformation
        try:
            numeric_features = ["no_of_adults", "no_of_children","no_of_weekend_nights","no_of_week_nights", "lead_time", "no_of_previous_cancellations","no_of_previous_bookings_not_canceled","avg_price_per_room","no_of_special_requests", "arrival_month"] # apply scaling
            categorical_features = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"] # apply one-hot encoding
            binary_features = ["required_car_parking_space", "repeated_guest"] # apply one-hot encoding with drop="if_binary"
            drop_features = ["arrival_year", "arrival_date"] # customers make bookings depending on mostly months  

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numeric_features}")
            logging.info(f"Binary columns: {binary_features}")
            logging.info(f"Dropped columns: {drop_features}")
            
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            binary_transformer = OneHotEncoder(drop="if_binary", dtype=int)

            preprocessor = make_column_transformer (
                (numeric_transformer, numeric_features),
                (categorical_transformer, categorical_features),
                (binary_transformer, binary_features),
                ("drop", drop_features),
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "booking_status"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)