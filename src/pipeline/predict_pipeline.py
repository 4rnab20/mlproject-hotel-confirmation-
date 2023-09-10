import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("output","model.pkl")
            preprocessor_path = os.path.join('output','preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        no_of_adults: int,
        no_of_weekend_nights: int,
        no_of_week_nights: int,
        required_car_parking_space: int,
        lead_time: float,
        no_of_special_requests: int,
        avg_price_per_room: float,
        arrival_month: int,
        market_segment_type: int,
        repeated_guest: int):

        self.no_of_adults = no_of_adults
        self.no_of_weekend_nights = no_of_weekend_nights
        self.no_of_week_nights = no_of_week_nights
        self.required_car_parking_space = required_car_parking_space
        self.lead_time = lead_time  
        self.no_of_special_requests = no_of_special_requests
        self.avg_price_per_room = avg_price_per_room
        self.arrival_month = arrival_month
        self.market_segment_type = market_segment_type
        self.repeated_guest = repeated_guest
        self.no_of_previous_cancellations = 0
        self.no_of_previous_bookings_not_canceled = 0
        self.type_of_meal_plan = 0
        self.room_type_reserved = 0
        self.arrival_date = 20
        self.arrival_year = 2018
        self.no_of_children = 0


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "no_of_adults": [self.no_of_adults],
                "no_of_weekend_nights": [self.no_of_weekend_nights],
                "no_of_week_nights": [self.no_of_week_nights],
                "required_car_parking_space": [self.required_car_parking_space],
                "lead_time": [self.lead_time],
                "no_of_special_requests": [self.no_of_special_requests],
                "avg_price_per_room": [self.avg_price_per_room],
                "arrival_month": [self.arrival_month],
                "market_segment_type": [self.market_segment_type],
                "repeated_guest": [self.repeated_guest],
                "no_of_previous_cancellations": [self.no_of_previous_cancellations],
                "no_of_previous_bookings_not_canceled": [self.no_of_previous_bookings_not_canceled],
                "type_of_meal_plan": [self.type_of_meal_plan],
                "room_type_reserved": [self.room_type_reserved],
                "arrival_date": [self.arrival_date],
                "arrival_year": [self.arrival_year],
                "no_of_children": [self.no_of_children]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        