import os
import sys
from src.bike.exception import CustomException
from src.bike.logger import logging
import pandas as pd
import chardet

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.bike.components.data_transformation import DataTransformation
from src.bike.components.data_transformation import DataTransformationConfig

from src.bike.components.model_tranier import ModelTrainerConfig
from src.bike.components.model_tranier import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Detect file encoding
            with open('D:\\Seoul Bike Share Demand Prediction\\data\\SeoulBikeData.csv', 'rb') as f:
                result = chardet.detect(f.read())
            encoding = result['encoding']

            # Read CSV with detected encoding
            df = pd.read_csv('D:\\Seoul Bike Share Demand Prediction\\data\\SeoulBikeData.csv', encoding=encoding)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))