import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from xgboost import XGBRegressor
import zipfile


def load_save_data(file_path_origin, file_path_destination):
    data = pd.read_csv(file_path_origin)

    data[['max_temp', 'sum_rain', 'sum_irr']] = data[['max_temp_lag1', 'sum_rain_lag1', 'sum_irr_lag1']].shift(-1)
    data = data[data['sensor_id'].isin([72, 76, 73, 74, 61, 63, 67, 65])].reset_index(drop=True)
    #data = data[data['sensor_id'].isin([72])].reset_index(drop=True)
    data = data[(data['date'] >= '2023-05-01') & (data['date'] <= '2023-08-31')]

    data = data[['sensor_id', 'date', 'avg_tens', 'max_temp', 'sum_rain', 'sum_irr', 'avg_tens_lag1', 'avg_tens_lag2',
                 'avg_tens_lag3', 'max_temp_lag1', 'max_temp_lag2', 'max_temp_lag3', 'sum_rain_lag1', 'sum_rain_lag2',
                 'sum_rain_lag3', 'sum_irr_lag1', 'sum_irr_lag2', 'sum_irr_lag3']]

    data.to_csv(file_path_destination, index=False)
    #data = data.drop(columns=['sensor_id', 'date']).sort_index(axis=1)

    return data


def save_zip_models(models, file_path):
    with zipfile.ZipFile(file_path, 'w') as zipf:
        for model_name, model in models.items():
            model_file = f"{model_name}.pkl"
            with open(model_file, 'wb') as file:
                pickle.dump(model, file)
            zipf.write(model_file, arcname=model_file)
            os.remove(model_file)


df = load_save_data('data/clean_data_rovere.csv', 'data/dss_data_rovere.csv')
full_df = df.drop(columns=['sensor_id', 'date']).sort_index(axis=1)

X_train, X_test, y_train, y_test = train_test_split(full_df.drop(columns=['avg_tens']), df['avg_tens'], test_size=0.2,
                                                    random_state=1)

# Linear Regression Model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

train_predictions = lr_model.predict(X_train)
train_error = np.sqrt(mean_squared_error(y_train, train_predictions))

cv_scores = cross_val_score(lr_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

test_predictions = lr_model.predict(X_test)
test_error = np.sqrt(mean_squared_error(y_test, test_predictions))

print('\n--- Linear Model ---')
print("Train Error:", round(train_error, 2))
print("Validation Error:", round(np.mean(cv_rmse_scores), 2))
print("Test Error:", round(test_error, 2))

# XGBoost Model

xgb_model = XGBRegressor(random_state=42)
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}

search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
search.fit(X_train, y_train)
xgb_model = search.best_estimator_

train_predictions = xgb_model.predict(X_train)
train_error = np.sqrt(mean_squared_error(y_train, train_predictions))

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

test_predictions = xgb_model.predict(X_test)
test_error = np.sqrt(mean_squared_error(y_test, test_predictions))

print('\n--- XG-Boost Model ---')
print("Train Error:", round(train_error, 2))
print("Validation Error:", round(np.mean(cv_rmse_scores), 2))
print("Test Error:", round(test_error, 2))


small_df = df[df['sensor_id'].isin([72])].reset_index(drop=True)
small_df = small_df.drop(columns=['sensor_id', 'date']).sort_index(axis=1)

X_train, X_test, y_train, y_test = train_test_split(small_df.drop(columns=['avg_tens']), small_df['avg_tens'], test_size=0.1,
                                                    random_state=1)

xgb_small_model = XGBRegressor(random_state=42)
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}

search = GridSearchCV(estimator=xgb_small_model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
search.fit(X_train, y_train)
xgb_small_model = search.best_estimator_


# Save Models

models = {
    'lr_model': lr_model,
    'xgb_model': xgb_model,
    'xgb_small_model': xgb_small_model
}
save_zip_models(models, 'src/dss_irrigation/mamdani/models_ml.zip')
