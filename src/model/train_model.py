import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from src.logging_config import logger
from src.model.model_define import (
    MYLSTM,
    MyDataset,
    convert_into_tensor_and_reshape,
    device,
)


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


def train_model(
    x_train: np.array,
    y_train: pd.array,
) -> MYLSTM:
    logger.info("Initializing model for training.")

    try:

        model = MYLSTM().to(device)

        train_dataset = MyDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
        )

        optimizer = optim.Adam(
            [
                *model.lstm1.parameters(),
                *model.fdPart.parameters(),
            ],
            lr=0.001,
        )

        lossfunc = nn.MSELoss()

        epochs = 20

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

        train_ds = load_dataset(data_path)

        logger.debug("Separating features and target.")

        X_train = train_ds.iloc[:, :-1].values
        y_train = train_ds.iloc[:, -1].values
        X_train,y_train=convert_into_tensor_and_reshape(X=X_train,y=y_train)
        logger.info(
            f"Training data prepared. X Shape: {X_train.shape}, y Shape: {y_train.shape}"
        )

        model = train_model(X_train, y_train)
        logger.info("========== Saving Model ==========")
        save_model(model_path,model)


        logger.info("========== Model Training Pipeline Completed Successfully ==========")

    except Exception:
        logger.exception("Model Training Pipeline Failed.")
        raise


if __name__ == "__main__":
    main()