import sys,os
import numpy as np
import pandas as pd
from src.constant import *
from src.logger import logging
from sklearn.impute import SimpleImputer
from src.exception import IncomeException
from src.entity.config_entity import DataTransformationConfig 
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from sklearn.preprocessing import LabelEncoder
from src.util.util import  read_json_file, save_data, save_model
from os import listdir





class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            

        except Exception as e:
            raise IncomeException(e,sys) from e
        


    def remove_unwanted_spaces(self, data)-> pd.DataFrame:
        try:
            data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            logging.info("Removed unwanted spaces in the dataframe")
            return data
        
        except Exception as e:
            raise IncomeException(e,sys) from e


    def remove_columns(self, data)-> pd.DataFrame:
        try:
            # Dropping the columns with all the null values
            null_columns = data.columns[data.isnull().all()]
            data.drop(null_columns, axis=1, inplace=True)

            json_path=self.data_transformation_config.schema_file_path
            json_info= read_json_file(json_path)
            columns=json_info['columns_to_remove']
         
            useful_data=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns

            useful_data.replace('?',np.NaN,inplace=True)

            useful_data[useful_data == ' ?'] = np.nan
            logging.info('Removing certain columns from the dataframe and replacing ? with Null values')
           
            return useful_data
        except Exception as e:
            raise IncomeException(e,sys) from e
        

    def impute_missing_values(self, data)-> pd.DataFrame:
        try:
            schema_file_path = self.data_transformation_config.schema_file_path

            dataset_schema = read_json_file(file_path=schema_file_path)

            numerical_columns = [x for x in dataset_schema[NUMERICAL_COLUMN_KEY] 
                                 if x not in dataset_schema[COLUMNS_TO_REMOVE] ]
            categorical_columns = [x for x in dataset_schema[CATEGORICAL_COLUMN_KEY] 
                                   if x not in dataset_schema[COLUMNS_TO_REMOVE] ]
            logging.info('Imputing numerical values')
            num_imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])
            
            logging.info('Imputing categorical values')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

            data['workclass'] = data['workclass'].apply(lambda x: 1 if 'Private' in x else 0)
            data['race'] = data['race'].apply(lambda x: 1 if 'White' in x else 0)
            data['country'] = data['country'].apply(lambda x: 1 if 'United-States' in x else 0)

            return data

        except Exception as e:
            raise IncomeException(e,sys) from e
        


    def encode_data(self, data)->pd.DataFrame:
        try:
                        
            schema_file_path = self.data_transformation_config.schema_file_path

            dataset_schema = read_json_file(file_path=schema_file_path)
            encoding_columns=dataset_schema[COLUMNS_FOR_ENCODING]
            logging.info('Label Eencoding for categorical data')

            encoder = LabelEncoder()
            for column in encoding_columns:
                if column in data.columns:
                    data[column] = encoder.fit_transform(data[column])

            return data

        except Exception as e:
            raise IncomeException(e,sys) from e

            
          
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining data file path.")
            schema_file_path = self.data_transformation_config.schema_file_path
            data_file_path = self.data_ingestion_artifact.data_file_path
            logging.info(f"Loading data as pandas dataframe.")

            #data_df = load_data(path=data_file_path, schema_file_path=schema_file_path)

            file_name = os.listdir(data_file_path)[0]
            file_path = os.path.join(data_file_path,file_name)

            data_df = pd.read_csv(file_path)

            schema = read_json_file(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]

            space_removed_data= self.remove_unwanted_spaces(data_df)
            columns_removed_data=self.remove_columns(space_removed_data)

            logging.info('Imputing missing values')
            imputed_data=self.impute_missing_values(columns_removed_data)

            logging.info('Encoding categorical values')
            encoded_data=self.encode_data(imputed_data)

            transformed_data_dir = self.data_transformation_config.transformed_data_dir
            data_file_name = 'Transformed_dataframe.csv'

            transformed_data_file_path = os.path.join(transformed_data_dir, data_file_name)
            logging.info(f"Saving transformed data.")
            save_data(transformed_data_file_path, encoded_data)

                 
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_data_file_path=transformed_data_dir
            )

            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise IncomeException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data transformation log completed.{'<<'*30} \n\n")