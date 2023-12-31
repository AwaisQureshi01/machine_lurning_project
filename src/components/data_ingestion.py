import os 
import sys
sys.path.insert(0,"../src")



# Add the project root directory to the Python path


from src.logger import logging
from src.exception import CustomException
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformations
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import *
from src.components.model_trainer import ModelTrainer


@dataclass
class Data_ingection_Config:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    row_data_path: str=os.path.join("artifacts","row.csv")

class Data_ingestion:
    def __init__(self):
        self.ingestion_config=Data_ingection_Config()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\stud.csv")
            logging.info("read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ =="__main__":
    obj= Data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformations()
    
    train_arr,test_arr,_ =data_transformation.initiate_data_transformation(train_data,test_data)

    ModelTrainer=ModelTrainer()
    ModelTrainer.initiate_model_trainer(train_arr,test_arr)
