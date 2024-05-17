import logging
import pandas as pd
from zenml import step
from src.model_dev import ClassificationModel

from zenml.client import Client
import mlflow
from keras.models import Model as KerasModel

from mlflow.keras import autolog as mlflow_keras_autolog
from keras.models import Sequential
experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train : pd.DataFrame, y_train : pd.Series, X_test : pd.DataFrame, y_test : pd.Series , l : int,vocab_size : int,max_length: int) -> Sequential:
    try:
        mlflow_keras_autolog()
        model = ClassificationModel(vocab_size,l,max_length)
        trained_model = model.train(X_train, y_train, X_test, y_test)

        # Save the model
        model_save_path = "/mnt/c/Users/hs414/OneDrive/Desktop/test/MLOPS_MAJOR/saved_models"
        trained_model.save("mymodel.keras")
        return trained_model
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
