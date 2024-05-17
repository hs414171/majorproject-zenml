from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"C:\Users\hs414\OneDrive\Desktop\Projects\MajorProject_MLOPS\data\data_encoded_main.csv")



#mlflow ui --backend-store-uri "file:C:\Users\hs414\AppData\Roaming\zenml\local_stores\732de32a-3c02-4138-86de-2bcb4445fcbe\mlruns"