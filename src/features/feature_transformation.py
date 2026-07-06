import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.logging_config import logger


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


def transform_X(X_train, X_test):
    logger.info("Applying MinMaxScaler on input features.")

    try:
        transformer = MinMaxScaler()

        transformer.fit(X_train.values.reshape(-1, 1))

        X_train = transformer.transform(
            X_train.values.reshape(-1, 1)
        ).reshape(X_train.shape)

        X_test = transformer.transform(
            X_test.values.reshape(-1, 1)
        ).reshape(X_test.shape)

        logger.info("Feature scaling completed successfully.")

        return transformer, X_train, X_test

    except Exception:
        logger.exception("Failed to scale input features.")
        raise


def transform_Y(y_train, y_test):
    logger.info("Applying MinMaxScaler on target values.")

    try:
        y_transformer = MinMaxScaler()

        y_transformer.fit(y_train.values.reshape(-1, 1))

        y_train = y_transformer.transform(
            y_train.values.reshape(-1, 1)
        )

        y_test = y_transformer.transform(
            y_test.values.reshape(-1, 1)
        )

        logger.info("Target scaling completed successfully.")

        return y_transformer, y_train, y_test

    except Exception:
        logger.exception("Failed to scale target values.")
        raise


def save_data(save_path: str, train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    logger.info("Saving transformed datasets.")

    try:
        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train_ds.csv")
        test_path = os.path.join(save_path, "test_ds.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.info("Datasets saved successfully.")

    except Exception:
        logger.exception("Failed to save transformed datasets.")
        raise


def save_transformers(
    X_transformer: MinMaxScaler,
    y_transformer: MinMaxScaler,
    save_dir: str,
) -> None:

    logger.info("Saving fitted transformers.")

    try:
        os.makedirs(save_dir, exist_ok=True)

        x_path = os.path.join(save_dir, "X_transformer.pkl")
        y_path = os.path.join(save_dir, "y_transformer.pkl")

        joblib.dump(X_transformer, x_path)
        joblib.dump(y_transformer, y_path)

        logger.info(f"X transformer saved at {x_path}")
        logger.info(f"Y transformer saved at {y_path}")

    except Exception:
        logger.exception("Failed to save transformers.")
        raise


def main() -> None:
    logger.info("========== Data Transformation Pipeline Started ==========")

    try:
        data_path = "data/interim"
        save_path = "data/processed"
        transformer_dir = "models"

        train_ds, test_ds = load_dataset(data_path)

        logger.debug("Separating features and target.")

        X_train = train_ds.iloc[:, :-1]
        y_train = train_ds.iloc[:, -1]

        X_test = test_ds.iloc[:, :-1]
        y_test = test_ds.iloc[:, -1]

        logger.info("Scaling feature matrix.")
        X_transformer, X_train, X_test = transform_X(X_train, X_test)

        logger.info("Scaling target vector.")
        y_transformer, y_train, y_test = transform_Y(y_train, y_test)

        logger.debug("Creating transformed datasets.")

        train_ds = pd.DataFrame(
            np.hstack((X_train, y_train.reshape(-1, 1)))
        )

        test_ds = pd.DataFrame(
            np.hstack((X_test, y_test.reshape(-1, 1)))
        )

        logger.info(
            f"Transformed Train Shape: {train_ds.shape}, "
            f"Test Shape: {test_ds.shape}"
        )

        save_data(
            save_path=save_path,
            train_ds=train_ds,
            test_ds=test_ds,
        )

        save_transformers(
            X_transformer=X_transformer,
            y_transformer=y_transformer,
            save_dir=transformer_dir,
        )

        logger.info("========== Data Transformation Pipeline Completed Successfully ==========")

    except Exception:
        logger.exception("Data Transformation Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()