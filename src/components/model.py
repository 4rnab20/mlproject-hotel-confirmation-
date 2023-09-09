import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("output","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
                "Random Forest" : {                 
                    'n_estimators': [50, 100, 200, 250],
                    'max_depth': [4, 8, 12],
                },
                "Gradient Boosting":{
                    'max_depth': [4, 8, 12],
                    'n_estimators': [50, 100, 200, 250]
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [50, 100, 200, 250]
                },
                "CatBoosting Classifier":{
                    'max_depth': [4, 8, 12],
                    'n_estimators': [50, 100, 200, 250]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [50, 100, 200, 250]
                }
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                             models = models, param = params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys()) [
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            # logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy, best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)