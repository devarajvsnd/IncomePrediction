import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import IncomeException
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
            ohe_columns= ["insured_occupation", "insured_relationship", "incident_type", "collision_type", "authorities_contacted"]
            encoded_data = pd.get_dummies(data, columns=ohe_columns)
            dump_col=['authorities_contacted_Ambulance','authorities_contacted_None', 
                'collision_type_Front Collision', 'incident_type_Multi-vehicle Collision', 
                'insured_occupation_adm-clerical', 'insured_relationship_husband']
            
            encoded_data=encoded_data.astype(float)

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            cluster_object_file_path =os.path.join(latest_model_dir, 'KMeans')
            file_name = os.listdir(cluster_object_file_path)[0]
            cluster_model = os.path.join(cluster_object_file_path,file_name) 
            kmeans=load_object(cluster_model)
            columns_used_for_clustering = kmeans.columns_used
            col_to_add=columns_used_for_clustering.difference(encoded_data.columns)
            for col in col_to_add:
                encoded_data[col] = 0

            dataframe = encoded_data.drop(columns=[col for col in dump_col if col in encoded_data.columns])
            df=dataframe[columns_used_for_clustering]
            df.astype(float) 
            cluster=kmeans.predict(df)
            model_name = find_correct_model_file(path=latest_model_dir, cluster_number=cluster[0])
            model_dir = os.path.join(latest_model_dir, model_name)
            file_name = os.listdir(model_dir)[0]
            
            model = load_object(os.path.join(model_dir, file_name))
            result=(model.predict(df))
            if result==0:
                prediction = "NO"
            else:
                prediction = "YES"
            return prediction
            
        except Exception as e:
            raise IncomeException(e, sys) from e

        