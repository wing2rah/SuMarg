import sys
import json
from pathlib import Path
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ml_engine.src.exception import CustomException
from ml_engine.src.logger import logging
from ml_engine.src.utils import save_object, evaluate_models
from traci_core.config import ROOT_DIR


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "model.pkl"
    metrics_file_path: Path = ROOT_DIR / "ml_engine" / "artifacts" / "metrics.json"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting transformed train and test arrays into X and y.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.01),

                "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),

                "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
                "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42),

                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
                "CatBoost": CatBoostRegressor(iterations=150, depth=6, learning_rate=0.1, verbose=0, random_seed=42),
                "LightGBM": LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)
            }

            logging.info(f"Initiating evaluation across {len(models)} candidate models via utils.evaluate_models.")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models=models)

            # Best model selection based on highest R2 Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["r2_score"])
            best_model_details = model_report[best_model_name]
            best_model_obj = best_model_details["model_obj"]

            if best_model_details["r2_score"] < 0.6:
                raise CustomException("No best model found with acceptable R2 score (> 0.6)", sys)

            logging.info(f"Best Model Selected: {best_model_name} with R2: {best_model_details['r2_score']:.4f}")

            # Save best model artifact (model.pkl)
            save_object(
                file_path=str(self.model_trainer_config.trained_model_file_path),
                obj=best_model_obj
            )

            # Save evaluation metrics (metrics.json)
            metrics_summary = {
                "best_model": best_model_name,
                "r2_score": best_model_details["r2_score"],
                "rmse": best_model_details["rmse"],
                "mae": best_model_details["mae"],
                "all_models": {
                    name: {k: v for k, v in info.items() if k != "model_obj"}
                    for name, info in model_report.items()
                }
            }

            with open(self.model_trainer_config.metrics_file_path, "w") as f:
                json.dump(metrics_summary, f, indent=4)

            logging.info(f"Trained model saved at: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"Metrics saved at: {self.model_trainer_config.metrics_file_path}")

            return best_model_name, best_model_details["r2_score"]

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from ml_engine.src.components.data_transformation import DataTransformation

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation()

    trainer = ModelTrainer()
    best_model, score = trainer.initiate_model_trainer(train_arr, test_arr)
    print("\n--- Pipeline Execution Summary ---")
    print(f"Status:     Training Succeeded")
    print(f"Best Model: {best_model}")
    print(f"R2 Score:   {score:.4f}")