import os
import sys
from src.bike.logger import logging
from src.bike.exception import CustomException
import pandas as pd

from src.bike.components.data_ingestion import DataIngestion
from src.bike.components.data_transformation import DataTransformation
from src.bike.components.model_tranier import ModelTrainer

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)