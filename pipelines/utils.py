import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataProcessingStrategy

import json



def get_data_for_test():
    try:
        df = pd.read_csv(r"/mnt/c/Users/hs414/OneDrive/Desktop/test/MLOPS_MAJOR/data/data_encoded_main.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataProcessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        X_train, X_test, y_train, y_test, label_encoder,vocab_size,max_length = data_cleaning.handle_data()
        x=X_train.tolist()
        result = json.dumps(x)
        return result
        
    except Exception as e:
        logging.error(e)
        raise e
    
