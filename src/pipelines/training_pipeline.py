import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from src.componets.data_ingestion  import DataIngestion 

from src.componets.data_transformation import DataTransformation
from src.componets.model_trainer import ModelTrainer
if __name__ == '__main__':

        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(train_data_path,test_data_path)
        
        data_transformation=DataTransformation()

        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        model_trainer.initiate_model_training(train_arr,test_arr)
