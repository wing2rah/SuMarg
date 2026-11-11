import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import yaml

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from traci_core.config import ROOT_DIR, DATASET_PROCESSED_DIR


@dataclass
class DataValidationConfig:
    schema_file_path: Path = ROOT_DIR / "ml_engine" / "config" / "schema.yaml"
    status_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "status.txt"
    train_data_path: Path = DATASET_PROCESSED_DIR / "train.csv"
    test_data_path: Path = DATASET_PROCESSED_DIR / "test.csv"


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def _read_schema(self) -> dict:
        try:
            with open(self.validation_config.schema_file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_all_columns(self) -> bool:
        logging.info("Data Validation component started.")
        try:
            validation_status = True
            schema = self._read_schema()
            expected_columns = list(schema.get("COLUMNS", {}).keys())

            if not self.validation_config.train_data_path.exists() or not self.validation_config.test_data_path.exists():
                raise FileNotFoundError("File not found")

            train_df = pd.read_csv(self.validation_config.train_data_path)
            test_df = pd.read_csv(self.validation_config.test_data_path)

            train_cols = list(train_df.columns)
            test_cols = list(test_df.columns)

            logging.info(f"Expected schema columns ({len(expected_columns)}): {expected_columns}")

            if len(train_cols) != len(expected_columns):
                validation_status = False
                logging.error(f"Train columns count mismatch! Expected {len(expected_columns)}, got {len(train_cols)}")

            if len(test_cols) != len(expected_columns):
                validation_status = False
                logging.error(f"Test columns count mismatch! Expected {len(expected_columns)}, got {len(test_cols)}")

            for col in expected_columns:
                if col not in train_cols:
                    validation_status = False
                    logging.error(f"Column [{col}] missing in Train dataset.")
                if col not in test_cols:
                    validation_status = False
                    logging.error(f"Column [{col}] missing in Test dataset.")

            self.validation_config.status_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_config.status_file_path, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")

            logging.info(f"Data Validation status: {validation_status}. Written to: {self.validation_config.status_file_path}")
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    validator = DataValidation()
    status = validator.validate_all_columns()
    print(f"\n[VALIDATION RESULT]: {'SUCCESS (True)' if status else 'FAILED (False)'}")