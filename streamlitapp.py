import streamlit as st
from typing import Union
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)




df = pd.read_csv(r"/mnt/c/Users/hs414/OneDrive/Desktop/test/MLOPS_MAJOR/data/data_encoded_main.csv")
X = df['Title'].tolist()
only_le = [word_tokenize(title.lower()) for title in X]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

categories = ['Accident', 'Aftermath', 'Animal', 'Beating', 'Beheading',
       'Bladed', 'Burning', 'Cartel', 'Combat', 'Compilation', 'Disaster',
       'Discussion', 'Drowning', 'Electricity', 'Execution', 'Explosions',
       'Falling', 'Fights', 'Gore', 'Industrial', 'Isis', 'Maiming',
       'Medical', 'Meta', 'Music', 'Other', 'Police', 'Request',
       'Shooting', 'Social', 'Suicide', 'Vehicle'] 




def remove_special_characters(token):
    return re.sub(r'[^\w\s]', '', token)

def transform_text(input_text):
    tokenized_text = word_tokenize(input_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokenized_text if word.lower() not in stop_words]
    cleaned_tokens = [remove_special_characters(token) for token in filtered_tokens]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
    transformed_text = ' '.join(stemmed_tokens)
    
    one_hot_result = tokenizer.texts_to_sequences([transformed_text])
    max_length = max(len(title) for title in only_le)
    one_hot_padded = pad_sequences(one_hot_result, maxlen=max_length, padding='pre')
    
    return one_hot_padded

# Streamlit app
def main():
    st.title("Title Categorization")

    # Text input
    user_input = st.text_area("Enter the title here:")

    if st.button("Transform"):
        if user_input:
            model_deployer = MLFlowModelDeployer.get_active_model_deployer()
            service = model_deployer.find_model_server(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                model_name="model",
                running = False
            )
            print(service[0])
            if service is None:
                st.write(
                    "No service could be found. The pipeline will be run first to create a service."
                )
            service[0].start(timeout=10)
            result = transform_text(user_input)
            pred = service[0].predict(result)   
            # Perform transformation
            
            
            st.write("Predicted_category: ")

            st.write(categories[pred.argmax()])
        else:
            st.write("Please enter some text to transform.")

if __name__ == "__main__":
    main()
