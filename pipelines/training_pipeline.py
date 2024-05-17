from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import evaluate_model
@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test,label_encoder,vocab_size,max_length = clean_df(df)
    print(vocab_size)
    model = train_model(X_train, y_train, X_test, y_test,label_encoder,vocab_size,max_length)
    accuracy = evaluate_model(model, X_test, y_test)
