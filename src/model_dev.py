from abc import ABC,abstractmethod

import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Reshape
import pandas as pd

class Model(ABC):
    def __init__(self, vocab_size, l):
        self.vocab_size = vocab_size
        self.l = l

    @abstractmethod
    def train(self,X_train, y_train, X_test, y_test):

        pass

class ClassificationModel(Model):
    def __init__(self, vocab_size,l,max_len):
        self.vocab_size = vocab_size
        self.l = l
        self.max_len = max_len
    def train(self, X_train, y_train, X_test, y_test):
        try:
            embedding_dim = 200
            X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

            model = Sequential()
            model.add(Embedding(self.vocab_size, embedding_dim, input_length=self.max_len))
            model.add(Reshape((self.max_len, embedding_dim)))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(128))
            model.add(Dense(self.l, activation='softmax'))

            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Use NumPy arrays for training
            model.fit(X_train_np, y_train_np, validation_data=(X_test_np, y_test_np), epochs=5, batch_size=32)

            
            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
