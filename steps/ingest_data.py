import logging
import pandas as pd
from zenml import step

class IngestData:
    """
        Ingesting Data from the path
    """
    def __init__(self,data_path:str) -> None:
        """
            Args : data_path : path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
            Returns : data : pd.DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step 
def ingest_data(data_path:str) -> pd.DataFrame:
    """
        Args : data_path : path to the data
        Returns : data : pd.DataFrame
    """
    try:
        df = IngestData(data_path).get_data()
        return df
    except Exception as e:
        logging.error(f"Error in ingesting the data from {data_path}: {e}")
        raise e


