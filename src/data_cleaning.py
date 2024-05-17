import logging
from abc import ABC,abstractmethod

import pandas as pd

from pandas.core.api import Series as Series

from typing import Union
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


def remove_special_characters(token):
    return re.sub(r'[^\w\s]', '', token)

class DataStrategy(ABC):
    """
        Abstract class that defines data strategy
    """

    @abstractmethod
    def handle_data(self,data:pd.DataFrame):
        """
            Args:
                data : data to be cleaned

            Returns:
                Union[pd.DataFrame,pd.series] -> X_train,y_train etc.
        """
        pass

class DataProcessingStrategy(DataStrategy):
    """
    Strategy to clean, process, and divide data
    """
    def handle_data(self, data: pd.DataFrame) :
        try:
            y = data['Category']
            X = data['Title'].tolist()
            
            # Clean and preprocess data
            tokenized_titles = [word_tokenize(title.lower()) for title in X]
            stop_words = set(stopwords.words('english'))
            filtered_titles = []
            for title_tokens in tokenized_titles:
                filtered_tokens = [word for word in title_tokens if word.lower() not in stop_words]
                filtered_titles.append(filtered_tokens)
            cleaned_titles = []
            for title_tokens in filtered_titles:
                title_without_special_chars = [remove_special_characters(token) for token in title_tokens]
                cleaned_titles.append(title_without_special_chars)
            stemmer = PorterStemmer()
            stemmed_titles = [' '.join([stemmer.stem(word) for word in title]) for title in cleaned_titles]
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X)
            one_hot_results = tokenizer.texts_to_sequences(stemmed_titles)
            max_length = max(len(title) for title in tokenized_titles)
            one_hot_padded = pad_sequences(one_hot_results, maxlen=max_length, padding='pre')
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(y)
            logging.info("Data preprocessing is successful")
            word_index = tokenizer.word_index
            vocab_size = len(word_index)+1


            # Divide data
            X_train, X_test, y_train, y_test = train_test_split(one_hot_padded, Y_encoded, test_size=0.2, random_state=42)
            l = len(label_encoder.classes_)
            return X_train, X_test, y_train, y_test, l,vocab_size,max_length

        except Exception as e:
            logging.error("Failed to process data: {}".format(e))
            raise e


class DataCleaning:
    """
    Class for preprocessing, dividing, and cleaning data
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        try:
            if isinstance(self.strategy, DataProcessingStrategy):
                X_train, X_test, y_train, y_test, l,vocab_size,max_length = self.strategy.handle_data(self.data)
                return X_train, X_test, y_train, y_test, l,vocab_size,max_length
            else:
                raise ValueError("Unsupported strategy type")

        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
