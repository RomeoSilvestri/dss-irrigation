# Path & Libraries

import sys
import os
import warnings

sys.path.append(os.path.abspath('dss'))
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import zipfile

from dss.package.dss_package import apply_decision_system, reverse_categorize_irrigation, show_figure


def process_irrigation_data(file_path, model_dss, model_simulation, dss, sensor_id, start_date, end_date):
    df = pd.read_csv(file_path)
    df = df[(df['sensor_id'] == sensor_id) & (df['date'] >= start_date) & (df['date'] <= end_date)]
    df.set_index('date', inplace=True)

    real_tens_values = df['avg_tens'][:-3]
    real_irrigation = df['sum_irr'][:-3]
    real_rain = df['sum_rain'][:-3]

    df['sum_irr'] = 0.0
    df = df.drop(columns=['sensor_id', 'avg_tens']).sort_index(axis=1)

    rec_irrigation = []
    rec_tens_values = []

    df['sum_irr_lag1'] = 0.0
    df['sum_irr_lag2'] = 0.0
    df['sum_irr_lag3'] = 0.0

    flag = 0
    for i in range(len(df) - 3):
        if i + 2 < len(df):
            predicted_rain_amount = df['sum_rain'].iloc[i] + df['sum_rain'].iloc[i + 1] + df['sum_rain'].iloc[i + 2]
        elif i + 1 < len(df):
            predicted_rain_amount = df['sum_rain'].iloc[i] + df['sum_rain'].iloc[i + 1]
        else:
            predicted_rain_amount = df['sum_rain'].iloc[i]

        new_df = pd.DataFrame({
            'last_avg_tensiometer': df['avg_tens_lag1'].iloc[i:i + 1],
            'predicted_avg_tensiometer': model_dss.predict(df.iloc[i:i + 1]),
            'predicted_rain_amount': predicted_rain_amount,
            'predicted_max_temperature': df['max_temp'].iloc[i:i + 1]
        })

        decision = apply_decision_system(dss, new_df.iloc[0:1])[0][1]

        if decision != 'Not Recommended':
            irrigation_value = reverse_categorize_irrigation(decision)
            flag = 1
        else:
            irrigation_value = 0.0

        df['sum_irr'].iloc[i] = irrigation_value
        rec_irrigation.append(irrigation_value)

        df['sum_irr_lag1'].iloc[i + 1] = irrigation_value
        df['sum_irr_lag2'].iloc[i + 2] = irrigation_value
        df['sum_irr_lag3'].iloc[i + 3] = irrigation_value

        if flag == 1:
            predicted_tens = max(10, model_simulation.predict(df.iloc[i:i + 1])[0])
            df['avg_tens_lag1'].iloc[i + 1] = predicted_tens
            df['avg_tens_lag2'].iloc[i + 2] = predicted_tens
            df['avg_tens_lag3'].iloc[i + 3] = predicted_tens
        else:
            predicted_tens = real_tens_values.iloc[i]

        rec_tens_values.append(predicted_tens)

    return real_tens_values, rec_tens_values, real_irrigation, rec_irrigation, real_rain


def compute_irr_statistics(real_tens_values, rec_tens_values, real_irrigation, rec_irrigation):
    real_count = sum(1 for value in real_irrigation if value > 0.0)
    real_sum = sum(real_irrigation)
    real_tens_days = sum(1 for value in real_tens_values if value > 400.0)

    rec_count = sum(1 for value in rec_irrigation if value > 0.0)
    rec_sum = sum(rec_irrigation)
    rec_tens_days = sum(1 for value in rec_tens_values if value > 400.0)

    return real_count, real_sum, real_tens_days, rec_count, rec_sum, rec_tens_days


def print_irr_statistics(real_count, real_sum, real_tens_days, rec_count, rec_sum, rec_tens_days):
    print("{:<20} {:<20} {:<20} {:<20}".format('', 'Over Threshold Days', 'Irrigation Days', 'Total Water (Liters)'))
    print("-" * 60)
    print("{:<20} {:<20} {:<20} {:<20}".format('Ground Truth', real_tens_days, real_count, int(real_sum)))
    print("{:<20} {:<20} {:<20} {:<20}".format('Recommended', rec_tens_days, rec_count, int(rec_sum)))


file_path = '../data/dss_data_rovere.csv'
start_date = '2023-05-01'
end_date = '2023-09-03'
sensor_id = 72

zip_path = 'models_ml.zip'
model_dss_name = 'global_lr.pkl'
model_simulation_name = f'sensor_{sensor_id}_lr.pkl'

with zipfile.ZipFile(zip_path, 'r') as zipf:
    with zipf.open(model_dss_name) as file:
        model_dss = pickle.load(file)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    with zipf.open(model_simulation_name) as file:
        model_simulation = pickle.load(file)

dss_file_path = 'model_dss.pkl'
with open(dss_file_path, 'rb') as file:
    dss = pickle.load(file)

real_tens_values, rec_tens_values, real_irrigation, rec_irrigation, real_rain = process_irrigation_data(file_path, model_dss, model_simulation, dss, sensor_id, start_date, end_date)
real_count, real_sum, real_tens_days, rec_count, rec_sum, rec_tens_days = compute_irr_statistics(real_tens_values, rec_tens_values, real_irrigation, rec_irrigation)

show_figure(sensor_id, real_tens_values, rec_tens_values, real_irrigation, rec_irrigation, real_rain, 'sensor_72.png')
print_irr_statistics(real_count, real_sum, real_tens_days, rec_count, rec_sum, rec_tens_days)