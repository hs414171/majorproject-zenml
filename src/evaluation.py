from abc import ABC, abstractmethod
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import Sequential
import pandas as pd
class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, model: Sequential, X_test: pd.DataFrame, y_test: pd.Series):
        pass

class EvaluateClassificationModel(Evaluation):
    def evaluate(self, model:Sequential, X_test: pd.DataFrame, y_test: pd.Series):
        try:
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            accuracy = accuracy_score(y_test, y_pred_classes)
            class_report = classification_report(y_test, y_pred_classes)
            conf_matrix = confusion_matrix(y_test, y_pred_classes)

            logging.info(f"Evaluation results - Accuracy: {accuracy:.4f}")
            logging.info(f"Classification Report:\n{class_report}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            return accuracy, class_report, conf_matrix

        except Exception as e:
            logging.error(f"Error in evaluating model: {e}")
            raise e
