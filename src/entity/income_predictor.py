import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import IncomeException
from sklearn.preprocessing import LabelEncoder
from src.util.util import  read_json_file, scale_numerical_columns, load_object, find_correct_model_file

class CensusData:

    def __init__(self,
                 age: float,
                 workclass:str,
                 fnlwgt:float,
                 education_num:float,
                 marital_status:str,
                 occupation:str,
                 race:str,
                 sex:str,
                 capital_gains:float,
                 capital_loss:float,
                 hours_per_week:float,
                 country:str
                 
                 ):
        try:
            self.age = age
            self.workclass = workclass
            self.fnlwgt = fnlwgt
            self.education_num = education_num
            self.marital_status = marital_status
            self.occupation = occupation
            self.race = race
            self.sex = sex
            self.capital_gains=capital_gains
            self.capital_loss=capital_loss
            self.hours_per_week=hours_per_week
            self.country=country
        except Exception as e:
            raise IncomeException(e, sys) from e

    def get_input_data_frame(self):

        try:
            input_dict = self.get_data_as_dict()
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise IncomeException(e, sys) from e

    def get_data_as_dict(self):
        try:
            input_data = {
                "age": [self.age],
                "workclass": [self.workclass],
                "fnlwgt": [self.fnlwgt],
                "education-num": [self.education_num],
                "marital-status": [self.marital_status],
                "occupation": [self.occupation],
                "race": [self.race],
                "sex": [self.sex],
                "capital-gain": [self.capital_gains],
                "capital-loss": [self.capital_loss],
                "hours-per-week": [self.hours_per_week],
                "country": [self.country]
                  
                }
            return input_data
        except Exception as e:
            raise IncomeException(e, sys)


class IncomePredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise IncomeException(e, sys) from e
        


    def predict(self, data):
        try:
            
            data['workclass'] = data['workclass'].apply(lambda x: 1 if 'Private' in x else 0)
            data['race'] = data['race'].apply(lambda x: 1 if 'White' in x else 0)
            data['country'] = data['country'].apply(lambda x: 1 if 'United-States' in x else 0)
            Columns_for_Encoding=["marital-status", "occupation", "sex"]

            encoder = LabelEncoder()
            for column in Columns_for_Encoding:
                data[column] = encoder.fit_transform(data[column])
            logging.info(f"Encoded input Form")

            encoded_data=data.astype(float)

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            model_name = os.listdir(latest_model_dir)[0]
            model_dir = os.path.join(latest_model_dir, model_name)
            file_name = os.listdir(model_dir)[0]
            
            model = load_object(os.path.join(model_dir, file_name))
            logging.info(f"Model Loaded")
            result=(model.predict(encoded_data))
            if result==0:
                prediction = "Less than 50K"
            else:
                prediction = "More than 50K"
            return prediction
            
        except Exception as e:
            raise IncomeException(e, sys) from e

        