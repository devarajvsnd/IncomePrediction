
from src.exception import IncomeException
import sys
from src.logger import logging
from typing import List
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.util.util import save_object,load_object, scale_numerical_columns, save_model
from src.entity.model_factory import MetricInfoArtifact, ModelFactory,GridSearchedBestModel
from src.entity.model_factory import evaluate_classification_model

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class FraudDetectionEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"




class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise IncomeException(e, sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_data_file_path = self.data_transformation_artifact.transformed_data_file_path

            file_name = os.listdir(transformed_data_file_path)[0]
            file_path = os.path.join(transformed_data_file_path,file_name)
            logging.info(f"Reading csv file: [{file_path}]")
            data=pd.read_csv(file_path)

            X=data.drop(['salary'],axis=1)
            Y=data['salary']
            train_acc={}
            test_acc={}
            model_acc={}

        
            x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=1 / 3, random_state=22)
        

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
        
        
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selecttion")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)
            
            logging.info(f"Best model found on training dataset: {best_model}")
            
            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list:List[GridSearchedBestModel]=model_factory.grid_searched_best_model_list
        
            model_list = [model.best_model for model in grid_searched_best_model_list ]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info:MetricInfoArtifact = evaluate_classification_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            logging.info(f"Best found model on both training and testing dataset.")
            model_object = metric_info.model_object


            trained_model_file_path=self.model_trainer_config.trained_model_file_path
            name=metric_info.model_name
            logging.info(f"Saving model at path: {trained_model_file_path}")

            save_model(trained_model_file_path, model_object, name)

            #Appending Train, test and Model accuracy for each clusters
            train_acc[name]=metric_info.train_accuracy
            test_acc[name]=metric_info.test_accuracy
            model_acc[name]=metric_info.model_accuracy
            


            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully", 
                                                            trained_model_file_path=trained_model_file_path, 
                                                            train_accuracy=train_acc, 
                                                            test_accuracy=test_acc, 
                                                            model_accuracy=model_acc)




            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

                
        except Exception as e:
            raise IncomeException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")