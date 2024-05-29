import os
import random
import numpy as np
import mlflow
import random as python_random
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""# Definindo funções adicionais"""

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""

def read_data():
    data = pd.read_csv(
        'https://raw.githubusercontent.com/DioniPinho/dataOps/main/fetal_health/datasoource/fetal_health_reduced.csv')
    x = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return x, y

read_data()

"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""

def process_data(x, y):
    global X_train, y_train
    columns_names = list(x.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(x)
    X_df = pd.DataFrame(X_df, columns=columns_names)
    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    y_train = y_train - 1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test

"""# 4 - Criando o modelo e adicionando as camadas"""
"""# 5 - Compilando o modelo

"""

def create_model(x):
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(input_shape=(x.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def config_mlflow():
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'dspmg'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e34f46a4e1cadccc86f37a0d58edb07c78f74aab'
    mlflow.set_tracking_uri('https://dagshub.com/dspmg/Healhty-test.mlflow')
    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)

"""# 6 - Executando o treino do modelo"""

def train_model(model, X_train, y_train, is_train=True ):
    with mlflow.start_run(run_name='experiment_mlops_ead') as run:
        model.fit(X_train,
                  y_train,
                  epochs=50,
                  validation_split=0.2,
                  verbose=3)

if __name__ == "__main__":
    x, y = read_data()
    X_train, X_test, y_train, y_test = process_data(x, y)
    model = create_model(x)
    config_mlflow()
    train_model(model, X_train, y_train)