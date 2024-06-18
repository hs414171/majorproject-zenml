import logging

import pandas as pd
import json

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


def remove_special_characters(token):
    return re.sub(r'[^\w\s]', '', token)



def get_data_for_test():
    try:
        df = pd.read_csv(r"/mnt/c/Users/hs414/OneDrive/Desktop/test/MLOPS_MAJOR/data/data_encoded_main.csv")
        df = df.sample(n=100)
        X = df['Title'].tolist()
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
        
        return one_hot_padded
        
    except Exception as e:
        logging.error(e)
        raise e
    
