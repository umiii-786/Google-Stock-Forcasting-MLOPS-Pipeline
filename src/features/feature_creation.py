import os
import yaml
import numpy as np
import pandas as pd
from src.logging_config import logger


def load_yaml(file_path: str) -> tuple:
    logger.info(f"Loading YAML file: {file_path}")

    try:
        with open(file_path) as f:
            params = yaml.safe_load(f)

        train_size = params["feature_creation"]["train_size"]
        last_n_features = params["feature_creation"]["last_n_features"]

        logger.info(
            f"Loaded parameters - train_size: {train_size}, last_n_features: {last_n_features}"
        )

        return train_size, last_n_features

    except Exception as e:
        logger.exception("Failed to load YAML file.")
        raise e


def load_dataset(data_path: str) -> pd.DataFrame:
    try:
        data_path = os.path.join(data_path, "data.csv")
        logger.info(f"Loading dataset from: {data_path}")

        df = pd.read_csv(data_path)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        return df

    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise e


def get_data_in_time_steps(last_n_features: int, series: pd.Series) -> tuple:
    logger.info("Creating time-step sequences.")

    try:
        X = []
        y = []

        for i in range((series.shape[0] - last_n_features) - 1):
            X.append(series[i:i + last_n_features])
            y.append(series[i + last_n_features])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Generated X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    except Exception as e:
        logger.exception("Failed while creating time-step sequences.")
        raise e


def split_data(train_size: float, X: np.ndarray, y: np.ndarray):
    logger.info("Splitting dataset into train and test sets.")

    try:
        train_size = int(len(X) * train_size)

        X_train = X[:train_size]
        X_test = X[train_size:]

        y_train = y[:train_size]
        y_test = y[train_size:]

        logger.info(
            f"Train shapes - X: {X_train.shape}, y: {y_train.shape}"
        )
        logger.info(
            f"Test shapes - X: {X_test.shape}, y: {y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception("Failed while splitting the dataset.")
        raise e


def save_data(save_path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        logger.info(f"Saving datasets to: {save_path}")

        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train_ds.csv")
        test_path = os.path.join(save_path, "test_ds.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.info("Train and Test datasets saved successfully.")

    except Exception as e:
        logger.exception("Failed to save datasets.")
        raise e


def main() -> None:
    logger.info("Feature creation pipeline started.")

    try:
        data_path = "data/raw"
        save_path = "data/interim"

        df = load_dataset(data_path)

        train_size, last_n_features = load_yaml("params.yaml")

        X, y = get_data_in_time_steps(
            last_n_features=last_n_features,
            series=df["Close"],
        )

        X_train, X_test, y_train, y_test = split_data(
            train_size=train_size,
            X=X,
            y=y,
        )

        train_ds = pd.DataFrame(
            np.hstack((X_train, y_train.reshape(-1, 1)))
        )

        test_ds = pd.DataFrame(
            np.hstack((X_test, y_test.reshape(-1, 1)))
        )

        logger.info(
            f"Train dataset shape: {train_ds.shape}, Test dataset shape: {test_ds.shape}"
        )

        save_data(
            save_path=save_path,
            train_ds=train_ds,
            test_ds=test_ds,
        )

        logger.info("Feature creation pipeline completed successfully.")

    except Exception as e:
        logger.exception("Feature creation pipeline failed.")
        raise e


if __name__ == "__main__":
    main()