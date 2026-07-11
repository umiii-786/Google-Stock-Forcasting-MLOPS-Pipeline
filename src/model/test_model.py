import os
import json
import torch
import joblib
import mlflow
from datetime import datetime
import numpy as np
import pandas as pd
from torch import nn
from dotenv import load_dotenv

from src.logging_config import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model.model_define import ( LSTM_Model, convert_into_tensor_and_reshape, device)

load_dotenv()
dagshub_pat=os.getenv("DAGSHUB_TOKEN")
if not dagshub_pat:
    raise EnvironmentError('DAGSHUB_PAT environment variable is not setted ') 
os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_pat 
os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_pat 

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/Google-Stock-Forcasting.mlflow/")

def load_dataset(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading processed datasets.")

    try:
        train_path = os.path.join(data_path, "train_ds.csv")
        test_path = os.path.join(data_path, "test_ds.csv")

        logger.debug(f"Train dataset path: {train_path}")
        logger.debug(f"Test dataset path: {test_path}")

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.info(
            f"Datasets loaded successfully. "
            f"Train Shape: {train_ds.shape}, Test Shape: {test_ds.shape}"
        )

        return train_ds, test_ds

    except FileNotFoundError:
        logger.error("Train or Test dataset file not found.")
        raise

    except Exception:
        logger.exception("Unexpected error while loading datasets.")
        raise


def load_params(file_path: str) -> dict:
    logger.info("Loading parameters from %s", file_path)

    try:
        with open(file_path, "r") as file:
            params = json.load(file)

        logger.info("Parameters loaded successfully.")
        return params

    except FileNotFoundError:
        logger.exception("Parameter file not found: %s", file_path)
        raise

    except json.JSONDecodeError:
        logger.exception("Invalid JSON format in: %s", file_path)
        raise

    except Exception:
        logger.exception("Unexpected error while loading parameters.")
        raise


def load_model(model_path: str,params:dict) -> LSTM_Model:
    logger.info("Loading trained model.")

    try:
        model_file = os.path.join(model_path, "model.pth")

        logger.debug(f"Model path: {model_file}")

        model = LSTM_Model(input_size=1,hidden_size=params['hidden_size'],
                           lstm_layers=params['lstm_layers'],
                           activation=params['activation'],
                           dense_layers=params['dense_layers'],
                           dense_units=params['dense_units'],
                           dropout=params['dropout']          
                 )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)

        logger.info("Model loaded successfully.")

        return model

    except FileNotFoundError:
        logger.error("Model file not found.")
        raise

    except Exception:
        logger.exception("Failed to load model.")
        raise


def load_y_transformer(model_dir: str) -> MinMaxScaler:
    logger.info("Loading y transformer.")

    try:
        y_path = os.path.join(model_dir, "y_transformer.pkl")

        logger.debug(f"Y transformer path: {y_path}")

        y_transformer = joblib.load(y_path)

        logger.info("Y transformer loaded successfully.")

        return y_transformer

    except FileNotFoundError:
        logger.error("Y transformer file not found.")
        raise

    except Exception:
        logger.exception("Failed to load y transformer.")
        raise

def load_ids(file_path: str) -> dict:
    logger.info("Loading MLflow IDs.")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        logger.info("MLflow IDs loaded successfully.")

        return data

    except FileNotFoundError:
        logger.error("JSON file containing IDs not found.")
        raise

    except Exception:
        logger.exception("Failed to load MLflow IDs.")
        raise

def test_model(
    model: LSTM_Model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    y_transformer: MinMaxScaler,
    run_id: str,
) -> dict:

    logger.info("Starting model evaluation.")

    try:
        model.eval()
        lossfunc = nn.MSELoss()

        with torch.no_grad():

            # ==========================
            # Training
            # ==========================
            logger.debug("Predicting on training dataset.")

            y_train_pred = model(X_train)
            train_loss = lossfunc(y_train_pred, y_train)

            # Convert to numpy and make sure arrays are 2D
            y_train_pred_np = y_train_pred.cpu().numpy().reshape(-1, 1)
            y_train_np = y_train.cpu().numpy().reshape(-1, 1)

            # Inverse transform
            real_y_train_pred = y_transformer.inverse_transform(y_train_pred_np)
            real_y_train = y_transformer.inverse_transform(y_train_np)

            train_rmse = np.sqrt(
                mean_squared_error(real_y_train, real_y_train_pred)
            )

            train_mae = mean_absolute_error(
                real_y_train,
                real_y_train_pred,
            )

            logger.info(
                f"Train Metrics -> "
                f"Loss: {train_loss.item():.6f}, "
                f"RMSE: {train_rmse:.4f}, "
                f"MAE: {train_mae:.4f}"
            )

            # ==========================
            # Testing
            # ==========================
            logger.debug("Predicting on test dataset.")

            y_test_pred = model(X_test)
            test_loss = lossfunc(y_test_pred, y_test)

            # Convert to numpy and make sure arrays are 2D
            y_test_pred_np = y_test_pred.cpu().numpy().reshape(-1, 1)
            y_test_np = y_test.cpu().numpy().reshape(-1, 1)

            # Inverse transform
            real_y_test_pred = y_transformer.inverse_transform(y_test_pred_np)
            real_y_test = y_transformer.inverse_transform(y_test_np)

            test_rmse = np.sqrt(
                mean_squared_error(real_y_test, real_y_test_pred)
            )

            test_mae = mean_absolute_error(
                real_y_test,
                real_y_test_pred,
            )

            logger.info(
                f"Test Metrics -> "
                f"Loss: {test_loss.item():.6f}, "
                f"RMSE: {test_rmse:.4f}, "
                f"MAE: {test_mae:.4f}"
            )

        # ==========================
        # Log to MLflow
        # ==========================
        logger.info(f"Logging metrics to MLflow Run ID: {run_id}")

        metrics = {
            "train_loss_scaled": train_loss.item(),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "test_loss_scaled": test_loss.item(),
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

        logger.info("Metrics logged successfully to MLflow.")

        return metrics

    except Exception:
        logger.exception("Model evaluation failed.")
        raise


def save_metrics_report(metrics: dict, report_dir: str = "reports", filename: str = "metrics_report.json"):

    try:
        # Create report directory if it doesn't exist

        report_path = os.path.join(report_dir, filename)

        # Add timestamp
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }

        # Save report
        with open(report_path, "a") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Metrics report successfully saved to '{report_path}'.")

    except Exception as e:
        logger.exception(f"Failed to save metrics report: {e}")
        raise


def main() -> None:
    logger.info("========== Model Evaluation Pipeline Started ==========")

    try:
        model_path = "models"
        params_path='reports/params.json'
        data_path = "data/processed"
        ids_path = os.path.join("reports", "data.json")
        params=load_params(params_path)
        model = load_model(model_path,params)

        y_transformer = load_y_transformer(model_path)

        ids = load_ids(ids_path)

        train_ds, test_ds = load_dataset(data_path)

        logger.debug("Preparing train tensors.")

        X_train = train_ds.iloc[:, :-1].values
        y_train = train_ds.iloc[:, -1].values

        X_train, y_train = convert_into_tensor_and_reshape(
            X_train,
            y_train,
        )

        logger.debug("Preparing test tensors.")

        X_test = test_ds.iloc[:, :-1].values
        y_test = test_ds.iloc[:, -1].values

        X_test, y_test = convert_into_tensor_and_reshape(
            X_test,
            y_test,
        )

        logger.info(
            f"Train Tensor Shape: {X_train.shape}, "
            f"Test Tensor Shape: {X_test.shape}"
        )

        metrics=test_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_transformer=y_transformer,
            run_id=ids["run_id"],
        )
        logger.info("========== Saving Metrics ==========")
        save_metrics_report(metrics=metrics)


        logger.info("========== Model Evaluation Pipeline Completed Successfully ==========")

    except Exception:
        logger.exception("Model Evaluation Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()