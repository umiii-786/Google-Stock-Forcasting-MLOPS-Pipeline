import os
import json
import torch
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch import nn, optim
from torch.utils.data import DataLoader
from src.logging_config import logger
from src.model.model_define import ( LSTM_Model, MyDataset, convert_into_tensor_and_reshape, device)

load_dotenv()
dagshub_pat=os.getenv("DAGSHUB_TOKEN")
if not dagshub_pat:
    raise EnvironmentError('DAGSHUB_PAT environment variable is not setted ') 
os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_pat 
os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_pat 

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/Google-Stock-Forcasting.mlflow/")

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


def load_dataset(data_path: str) -> pd.DataFrame:
    logger.info("Loading training dataset.")

    try:
        data_path = os.path.join(data_path, "train_ds.csv")
        logger.debug(f"Dataset path: {data_path}")

        df = pd.read_csv(data_path)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"Dataset not found: {data_path}")
        raise

    except Exception:
        logger.exception("Unexpected error while loading dataset.")
        raise


def train_model(x_train: np.array, y_train: pd.array,params:dict) -> LSTM_Model:
    logger.info("Initializing model for training.")

    try:

        model = LSTM_Model(input_size=1,hidden_size=params['hidden_size'],
                           lstm_layers=params['lstm_layers'],
                           activation=params['activation'],
                           dense_layers=params['dense_layers'],
                           dense_units=params['dense_units'],
                           dropout=params['dropout']          
                 ).to(device)

        train_dataset = MyDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'])

        optimizer = optim.Adam(
            [
                *model.lstm.parameters(),
                *model.fc.parameters(),
            ],
            lr=params['lr'],
        )

        lossfunc = nn.MSELoss()

        epochs = params['epochs']

        logger.info(
            f"Training started for {epochs} epochs on device: {device}"
        )

        for epoch in range(epochs):

            model.train()
            epoch_loss = 0.0

            for batch, target in train_loader:

                batch = batch.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                y_pred = model(batch)

                loss = lossfunc(y_pred, target)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)

            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}"
            )

        logger.info("Model training completed successfully.")

        return model

    except Exception:
        logger.exception("Error occurred during model training.")
        raise


def log_params_and_model(
    model: torch.nn.Module,
    params: dict,
    experiment_name: str,
    input_example: torch.Tensor
) -> tuple[str, str]:

    logger.info("Logging model and parameters to MLflow...")

    try:

        mlflow.set_experiment(experiment_name)

        model.eval()

        with mlflow.start_run() as run:

            # Parameters
            mlflow.log_params(params)

            # Artifact
            mlflow.log_artifact(
                "reports/figures/train_test_forcast_graph_results.png",
                artifact_path="plots"
            )

            # Model
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="lstm_model",
                serialization_format="pickle",   # <-- important
                input_example=input_example.cpu()
            )

            model_id = model_info.model_id
            run_id = run.info.run_id

            logger.info("Run ID: %s", run_id)
            logger.info("Model ID: %s", model_id)

            return model_id, run_id

    except Exception:
        logger.exception("Failed to log model.")
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

        file_path = os.path.join('reports', 'data.json')

        with open(file_path, "w") as f:
            json.dump(ids, f, indent=4)

        logger.info(f"IDs saved successfully at {file_path}")

    except Exception:
        logger.exception("Error occurred while saving IDs")
        raise

def save_model(model_path: str, model: torch.nn.Module) -> None:
    logger.info("Saving trained model.")

    try:
        os.makedirs(model_path, exist_ok=True)

        model_file = os.path.join(model_path, "model.pth")

        torch.save(model.state_dict(), model_file)

        logger.info(f"Model saved successfully at: {model_file}")

    except Exception:
        logger.exception("Failed to save trained model.")
        raise


def main() -> None:
    logger.info("========== Model Training Pipeline Started ==========")

    try:
        model_path='models'
        data_path = "data/processed"
        params_path='reports/params.json'
       
        params = load_params(params_path)
        train_ds = load_dataset(data_path)

        logger.debug("Separating features and target.")

        X_train = train_ds.iloc[:, :-1].values
        y_train = train_ds.iloc[:, -1].values
        X_train,y_train=convert_into_tensor_and_reshape(X=X_train,y=y_train)
        logger.info(
            f"Training data prepared. X Shape: {X_train.shape}, y Shape: {y_train.shape}"
        )

        model = train_model(X_train, y_train,params=params)
        logger.info("========== Tracking Parameter and Logging Model with MLFLOW ==========")
        experiment_name="Trained_Model_Google_Stock_forcasting"
        model_id, run_id = log_params_and_model(
            model=model,
            params=params,
            input_example=X_train[:5],      # first five samples
            experiment_name=experiment_name
        )


        logger.info("========== Saving Model ==========")
        save_model(model_path,model)

        logger.info("========== Saving ID ==========")
        save_ids(model_id=model_id,run_id=run_id,model_name='model')


        logger.info("========== Model Training Pipeline Completed Successfully ==========")

    except Exception:
        logger.exception("Model Training Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()