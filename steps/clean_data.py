import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataProcessingStrategy
from src.data_cleaning import DataCleaning, DataProcessingStrategy
import pandas as pd
from zenml import step
from typing import Tuple
import numpy as np
@step
def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int,int,int]:
    """Clean and preprocess the data"""
    try:
        # Use the new DataProcessingStrategy
        processing_strategy = DataProcessingStrategy()
        data_cleaning = DataCleaning(df, processing_strategy)
        X_train, X_test, y_train, y_test, label_encoder,vocab_size,max_length = data_cleaning.handle_data()
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        logging.info("Data cleaning and preprocessing completed")
        return X_train_df, X_test_df, y_train_series, y_test_series, label_encoder,vocab_size,max_length

    except Exception as e:
        logging.error("Error in cleaning and preprocessing data: {}".format(e))
        raise e
