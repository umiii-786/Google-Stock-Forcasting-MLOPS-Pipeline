import os 
import json 
import mlflow
import pandas as pd 
from mlflow import MlflowClient
from src.logging_config import logger

dagshub_pat=os.getenv("DAGSHUB_TOKEN")

if not dagshub_pat:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_pat
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_pat

mlflow.set_tracking_uri("https://dagshub.com/umiii-786/Google-Stock-Forcasting.mlflow/")



# ----------------- Load run IDs from JSON -----------------
def get_ids(ids_path: str):
    try:
        logger.info(f"Loading run/model IDs from: {ids_path}")

        with open(ids_path, 'r') as file:
            data = json.load(file)

        logger.info(f"IDs loaded successfully: {data}")
        return data

    except Exception:
        logger.exception("Error occurred while loading IDs JSON")
        raise

def register_model_new(model_id: str,
                       model_name: str,
                       reg_model_name: str):

    try:
        logger.info("Starting model registration")

        # Same URI format as your previous project
        model_uri = f"models:/{model_id}"

        logger.info(f"Model URI: {model_uri}")

        mv = mlflow.register_model(
            model_uri=model_uri,
            name=reg_model_name
        )

        logger.info(
            f"Registered model '{mv.name}' Version {mv.version}"
        )

        return mv.name, mv.version

    except Exception:
        logger.exception("Error while registering model")
        raise


def main()->None:
    registered_model_name = "Google_Stock_Forecast_Model"
    ids_path='reports/data.json'
    meta_data=get_ids(ids_path=ids_path)
    register_model_new(model_id=meta_data['model_id'],model_name=meta_data['model_name'],
                       reg_model_name=registered_model_name
                       )


if __name__=='__main__':
    main()

