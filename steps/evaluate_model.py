import logging
import pandas as pd
from zenml import step
from src.evaluation import EvaluateClassificationModel
from typing_extensions import Annotated

from zenml.client import Client
import mlflow
from tensorflow.keras.models import Model as KerasModel
experiment_tracker = Client().active_stack.experiment_tracker
from tensorflow.keras.models import Sequential

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model:Sequential,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    try:

        evaluate_class = EvaluateClassificationModel()
        acc,class_report,confusion_matrix = evaluate_class.evaluate(model,X_test,y_test)
        mlflow.log_metric("accuracy",acc)
        logging.info("accuracy : {}".format(acc))

        return acc
    
    except Exception as e:
        logging.error("Error in evaluating model {}".format(e))
        raise e