import os
import sys
from src.exception import CustomException   # Custom exception handling
from src.logger import logging             # Custom logging module
import pandas as pd

from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass                     # For config class

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# Configuration class to store file paths
@dataclass   #is a decorator that automatically generates special methods for a class that mainly stores data.eg(__init__())
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
        


# This ensures the code runs only when this file is executed directly
if __name__ == "__main__":

    #  Create an object of DataIngestion class
    # This class is responsible for loading and splitting the dataset
    obj = DataIngestion()

    # Call the data ingestion method
    # It returns training and testing datasets
    train_data, test_data = obj.initate_data_ingestion()  # (typo likely: initiate_data_ingestion)

    #  Create an object of DataTransformation class
    # This class handles preprocessing like encoding, scaling, cleaning, etc.
    data_Transformation = DataTransformation()

    # Transform the raw train and test data into model-ready format (arrays)
    # _ is used to ignore extra returned value (like preprocessing pipeline)
    train_arr, test_arr, _ = data_Transformation.initiate_data_transformation(
        train_data, test_data
    )

    #  Create an object of ModelTrainer class
    # This class is responsible for training and evaluating the ML model
    modeltrainer = ModelTrainer()

    # Train the model using transformed data and print evaluation results
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))