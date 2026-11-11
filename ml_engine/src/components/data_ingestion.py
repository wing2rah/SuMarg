import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import DATASET_RAW_PATH, DATASET_PROCESSED_DIR


@dataclass
class DataIngestionConfig:
    raw_data_path: Path = DATASET_RAW_PATH
    processed_dir: Path = DATASET_PROCESSED_DIR
    train_data_path: Path = DATASET_PROCESSED_DIR / "train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "test.csv"
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion component started.")
        try:
            if not self.ingestion_config.raw_data_path.exists():
                raise FileNotFoundError(
                    f"Raw dataset file not found at: {self.ingestion_config.raw_data_path}"
                )

            logging.info(f"Reading raw data from: {self.ingestion_config.raw_data_path}")
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw data shape: {df.shape}")

            self.ingestion_config.processed_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Splitting data: test_size = {self.ingestion_config.test_size}")
            train_df, test_df = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                shuffle=True
            )

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved: {self.ingestion_config.train_data_path} ({len(train_df)} rows)")
            logging.info(f"Test data saved: {self.ingestion_config.test_data_path} ({len(test_df)} rows)")
            logging.info("Data Ingestion completed successfully.")

            return str(self.ingestion_config.train_data_path), str(self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"\n[SUCCESS] Train file: {train_path}")
    print(f"[SUCCESS] Test file: {test_path}")