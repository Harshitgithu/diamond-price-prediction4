import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

# Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            # Read the dataset
            df = pd.read_csv(os.path.join('notebook/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved')

            # Perform train-test split
            logging.info("Train-test split")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train and test sets saved')

            logging.info('Ingestion of data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error('Error occurred in Data Ingestion process', exc_info=True)
            raise CustomException(e, sys)
