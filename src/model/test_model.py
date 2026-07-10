import os 
import torch
from torch import nn
from model_define import MYLSTM
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.logging_config import logger
from sklearn.preprocessing import MinMaxScaler
import pickle
from src.model.model_define import convert_into_tensor_and_reshape
import json
import mlflow


def load_dataset(data_path: str) -> tuple:
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


def load_model(model_path):
    model_path=os.path.join(model_path,'model.pth')
    model = MYLSTM()
    model.load_state_dict(torch.load(model_path))
    return model


def load_y_transformer(
    model_dir: str,
) -> MinMaxScaler:
    logger.info("Loading fitted transformers.")

    try:
        y_path = os.path.join(model_dir, "y_transformer.pkl")

        logger.debug(f"Y transformer path: {y_path}")

        y_transformer = pickle.load(y_path)

        logger.info("Y transformer loaded successfully.")

        return y_transformer

    except FileNotFoundError:
        logger.error("Transformer file not found.")
        raise

    except Exception:
        logger.exception("Failed to load transformer.")
        raise


def log_model_and_parameters(model, parameters, signature):
    try:
        logger.info("Starting MLflow logging")

        mlflow.set_experiment(experiment_name='Pipeline Result')

        with mlflow.start_run() as run:
            logger.info(f"MLflow run started with run_id: {run.info.run_id}")

            # Log parameters
            mlflow.log_params(parameters)
            logger.info("Parameters logged successfully")

            # Log model
            logged_model =mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model"
                )
            logger.info("Model logged successfully!")
            run_id=run.info.run_id
            model_id = logged_model.model_id
            model_name='model'
            logger.info(f"run ID:{run_id} model ID: {model_id}, Model name: {model_name}")

            return model_id, run_id,model_name

    except Exception:
        logger.exception("Error occurred during MLflow logging")
        raise


def save_ids(model_id: str, model_name: str,run_id:str):
    try:
        logger.info("Saving run_id and model_id to JSON")

        os.makedirs('reports', exist_ok=True)

        ids = {
            'model_name': model_name,
            'model_id': model_id,
            'run_id': run_id,
        }

        file_path = os.path.join('reports', 'ids.json')

        with open(file_path, "w") as f:
            json.dump(ids, f, indent=4)

        logger.info(f"IDs saved successfully at {file_path}")

    except Exception:
        logger.exception("Error occurred while saving IDs")
        raise


def test_model(model,X_train,y_train,X_test,y_test,y_transformer):
    model.eval()
    lossfunc=nn.MSELoss()
    with torch.no_grad():
        # Training Prediction
        # ==========================
        y_train_pred = model(X_train)

        # Loss on scaled data
        train_loss = lossfunc(y_train_pred, y_train)
        print(f"Scaled Train Loss : {train_loss.item():.6f}")

        # Convert to numpy
        y_train_pred_np = y_train_pred.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        # Inverse scaling
        real_y_train_pred = y_transformer.inverse_transform(y_train_pred_np)
        real_y_train = y_transformer.inverse_transform(y_train_np)

        # Metrics on original scale
        train_rmse = np.sqrt(mean_squared_error(real_y_train, real_y_train_pred))
        train_mae = mean_absolute_error(real_y_train, real_y_train_pred)

        print(f"Train RMSE : {train_rmse:.4f}")
        print(f"Train MAE  : {train_mae:.4f}")


        # ==========================
        # Test Prediction
        # ==========================
        y_test_pred = model(X_test)

        # Loss on scaled data
        test_loss = lossfunc(y_test_pred, y_test)
        print(f"Scaled Test Loss : {test_loss.item():.6f}")

        # Convert to numpy
        y_test_pred_np = y_test_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Inverse scaling
        real_y_test_pred = y_transformer.inverse_transform(y_test_pred_np)
        real_y_test = y_transformer.inverse_transform(y_test_np)

        # Metrics on original scale
        test_rmse = np.sqrt(mean_squared_error(real_y_test, real_y_test_pred))
        test_mae = mean_absolute_error(real_y_test, real_y_test_pred)

        print(f"Test RMSE : {test_rmse:.4f}")
        print(f"Test MAE  : {test_mae:.4f}")

def main()->None:
    model_path='models'
    data_path='data/processed'
    model=load_model(model_path=model_path)
    y_transformer=load_y_transformer(model_dir=model)
    train_ds, test_ds=load_dataset(data_path=data_path)
    X_train = train_ds.iloc[:, :-1].values
    y_train = train_ds.iloc[:, -1].values
    X_train,y_train=convert_into_tensor_and_reshape(X_train,y_train)

    X_test = test_ds.iloc[:, :-1].values
    y_test = test_ds.iloc[:, -1].values
    X_test,y_test=convert_into_tensor_and_reshape(X_test,y_test)
    
    test_model(model,X_train,y_train,X_test,y_test,y_transformer)

if __name__=='__main__':
    main()


    




