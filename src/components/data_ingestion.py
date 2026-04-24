import os
import sys
from src.exception import CustomException   # Custom exception handling
from src.logger import logging             # Custom logging module
import pandas as pd

from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass                     # For config class


# Configuration class to store file paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path for training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')    # Path for testing data
    data_path: str = os.path.join('artifacts', 'data.csv')         # Path for raw data


# Main Data Ingestion class
class DataIngestion:
    def __init__(self):
        # Initialize config object
        self.ingestion_conf = DataIngestionConfig()

    def initate_data_ingestion(self):
        # Log start of ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            # Read dataset from CSV file
            df = pd.read_csv(r"notebook\data\student.csv")

            # Create directory 'artifacts' if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_conf.train_data_path), exist_ok=True)

            # Save raw dataset into artifacts folder
            df.to_csv(self.ingestion_conf.data_path, index=False, header=True)

            # Log before splitting dataset
            logging.info("Train test split initiated")

            # Split dataset into training (80%) and testing (20%)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training dataset
            train_set.to_csv(self.ingestion_conf.train_data_path, index=False, header=True)

            # Save testing dataset
            test_set.to_csv(self.ingestion_conf.test_data_path, index=False, header=True)

            # Log completion of ingestion
            logging.info("Ingestion of the data is completed")

            # Return paths of saved files
            return (
                self.ingestion_conf.train_data_path,
                self.ingestion_conf.test_data_path,
            )

        except Exception as e:
            # Raise custom exception if any error occurs
            raise CustomException(e, sys)