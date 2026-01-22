import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import zipfile

def load_save_data(file_path_origin, file_path_destination):
    data = pd.read_csv(file_path_origin)

    data[['max_temp', 'sum_rain', 'sum_irr']] = data[['max_temp_lag1', 'sum_rain_lag1', 'sum_irr_lag1']].shift(-1)
    data = data[data['sensor_id'].isin([72, 76, 73, 74, 61, 63, 67, 65])].reset_index(drop=True)
    data = data[(data['date'] >= '2023-05-01') & (data['date'] <= '2023-08-31')]

    data = data[['sensor_id', 'date', 'avg_tens', 'max_temp', 'sum_rain', 'sum_irr', 'avg_tens_lag1', 'avg_tens_lag2',
                 'avg_tens_lag3', 'max_temp_lag1', 'max_temp_lag2', 'max_temp_lag3', 'sum_rain_lag1', 'sum_rain_lag2',
                 'sum_rain_lag3', 'sum_irr_lag1', 'sum_irr_lag2', 'sum_irr_lag3']]

    data.to_csv(file_path_destination, index=False)

    return data

def save_zip_models(models, file_path):
    with zipfile.ZipFile(file_path, 'w') as zipf:
        for model_name, model in models.items():
            model_file = f"{model_name}.pkl"
            with open(model_file, 'wb') as file:
                pickle.dump(model, file)
            zipf.write(model_file, arcname=model_file)
            os.remove(model_file)

def train_and_save_models(data, output_zip):
    models = {}

    # Modelli globali
    global_df = data.drop(columns=['sensor_id', 'date']).sort_index(axis=1)
    X = global_df.drop(columns=['avg_tens'])
    y = global_df['avg_tens']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # Linear Regression globale
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['global_lr'] = lr_model

    # XGBoost globale
    xgb_model = XGBRegressor(random_state=42)
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }
    search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    models['global_xgb'] = search.best_estimator_

    # Modelli specifici per ciascun tensiometro
    for sensor_id in data['sensor_id'].unique():
        sensor_df = data[data['sensor_id'] == sensor_id].drop(columns=['sensor_id', 'date']).sort_index(axis=1)
        X = sensor_df.drop(columns=['avg_tens'])
        y = sensor_df['avg_tens']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        # Linear Regression specifico
        sensor_lr_model = LinearRegression()
        sensor_lr_model.fit(X_train, y_train)
        models[f'sensor_{sensor_id}_lr'] = sensor_lr_model

        # XGBoost specifico
        sensor_xgb_model = XGBRegressor(random_state=42)
        search = GridSearchCV(estimator=sensor_xgb_model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X_train, y_train)
        models[f'sensor_{sensor_id}_xgb'] = search.best_estimator_

    # Salvataggio dei modelli
    save_zip_models(models, output_zip)

# Caricamento e preprocessing dei dati
df = load_save_data('../data/clean_data_rovere.csv', '../data/dss_data_rovere.csv')

# Allenamento e salvataggio dei modelli
train_and_save_models(df, 'models_ml.zip')
