import yaml
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from src.logging_config import logger
import os 

def load_yaml(file_path: str) -> str:
    logger.info(f"Loading YAML file: {file_path}")

    with open(file_path) as f:
        params = yaml.safe_load(f)

    filename = params["data_ingestion"]["filename"]
    logger.info(f"Filename loaded from YAML: {filename}")

    return filename


# Load the latest version
def load_dataset(filename: str) -> pd.DataFrame:
    logger.info(f"Loading dataset: {filename}")

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "henryshan/google-stock-price",
        filename,
    )

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df

def save_data(save_path: str,df:pd.DataFrame):
    try:
        logger.info(f"Saving datasets to path: {save_path}")

        os.makedirs(save_path, exist_ok=True)

        data_path = os.path.join(save_path, 'data.csv')

        df.to_csv(data_path, index=False)

        logger.info(" datasets saved successfully")

    except Exception as e:
        logger.exception("Error occurred while saving data")
        raise e




def main() -> None:
    logger.info("Data ingestion pipeline started.")

    try:
        yaml_path = "params.yaml"

        filename = load_yaml(yaml_path)
        df = load_dataset(filename)
        
        save_path='data/raw'
        logger.info("Saving Dataset at ",save_path)
        save_data(save_path=save_path,df=df)

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()