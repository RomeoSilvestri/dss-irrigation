# Path & Libraries

import sys
import os
import warnings

sys.path.append(os.path.abspath('src'))
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import zipfile

from dss_irrigation.mamdani.dss_package import (select_data_ml, load_weather_data, combine_data_dss, load_true_irrigation,
                                     multiple_recommendation)  # Custom Package

# Parameters of Interest

file_path = 'data/dss_data_rovere.csv'
current_date = '2023-06-20'
days_predicted = 3

sensor_id = 72
latitude, longitude = 46.245079, 11.164434


# Data Loading & Processing

# 1) CSV File with Rules & Features Definition

rules_excel_path = 'data/dss_rules.xlsx'
sheet_names = ['Rules', 'Crispy Values', 'Fuzzy Values']

rules_df, crispy_df, fuzzy_df = [pd.read_excel(rules_excel_path, sheet_name=sheet) for sheet in sheet_names]
rules_df = rules_df.iloc[:, :].rename(columns=lambda x: x.lower().replace(' ', '_'))
fuzzy_df = fuzzy_df.iloc[:, :].rename(columns=lambda x: x.lower().replace(' ', '_'))

# 2) Past Tensiometer & Weather Data (Last 3 Days)
data_ml = select_data_ml(file_path, sensor_id, current_date, days_predicted)

# 3) Weather Forecast Data from OpenMeteo API (Next 3 Days)
weather_set = load_weather_data(latitude, longitude, current_date, days_predicted)

# 4) Final Dataset for DSS
data_dss = combine_data_dss(data_ml, weather_set)  # it combines the data from the previous two steps

# 5) ML Model for Predicting Tensiometer Value for the Next Day (it could be used iteratively)

zip_path = 'src/dss_irrigation/mamdani/models_ml.zip'
model_name = 'global_xgb.pkl'
with zipfile.ZipFile(zip_path, 'r') as zipf:
    with zipf.open(model_name) as file:
        xgb_model = pickle.load(file)

# 6) Real Irrigation Data for the Next 3 Days (OPTIONAL)
true_irrigation = load_true_irrigation(file_path, sensor_id, current_date, days_predicted)

# 7) Validation Data to optimize the DSS (OPTIONAL)

# val_csv_path = '../data/boost_val_data.csv'
# data_validation = pd.read_csv(val_csv_path)


# DSS Architecture

# Fuzzy Input Variables (Universe & Membership Functions)

input_variables = rules_df.columns[:-1].tolist()
crispy_input = crispy_df.iloc[:, 0].dropna().values
crispy_output = crispy_df.iloc[:, 1].dropna().values

fuzzy_values = fuzzy_df.values
fuzzy_lower = fuzzy_values[0].astype(float)
fuzzy_min = fuzzy_values[1].astype(float)
fuzzy_max = fuzzy_values[2].astype(float)
fuzzy_upper = fuzzy_values[3].astype(float)
num_classes = fuzzy_values[4].astype(float)

feature_dict = {}
universe_dict = {}

for var, lower, min_val, max_val, upper, num_class in zip(input_variables, fuzzy_lower[:-1], fuzzy_min[:-1],
                                                          fuzzy_max[:-1], fuzzy_upper[:-1], num_classes[:-1]):
    feature_dict[var] = {}
    fraction_range = (max_val - min_val) / (num_class - 1)
    universe = np.arange(lower, upper + 1, 1)
    universe_dict[var] = universe
    for i, term_name in enumerate(crispy_input):
        if i == 0:
            feature_dict[var] = ctrl.Antecedent(universe, var)
            term_range = [lower, lower, min_val, min_val + fraction_range]
            feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
        elif i == num_class - 1:
            term_range = [max_val - fraction_range, max_val, upper, upper]
            feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
        else:
            term_range = [min_val + (i - 1) * fraction_range, min_val + i * fraction_range,
                          min_val + (i + 1) * fraction_range]
            feature_dict[var][term_name] = fuzz.trimf(feature_dict[var].universe, term_range)

# Fuzzy Output Variable

output_variable = rules_df.columns[-1]

var = output_variable
lower = fuzzy_lower[-1].astype(float)
min_val = fuzzy_min[-1].astype(float)
max_val = fuzzy_max[-1].astype(float)
upper = fuzzy_upper[-1].astype(float)
num_class = num_classes[-1].astype(float)

feature_dict[var] = {}
fraction_range = (max_val - min_val) / (num_class - 1)
universe = np.arange(lower, upper + 1, 1)
universe_dict[var] = universe
for i, term_name in enumerate(crispy_output):
    if i == 0:
        feature_dict[var] = ctrl.Consequent(universe, var)
        term_range = [lower, lower, min_val, min_val + fraction_range]
        feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
    elif i == num_class - 1:
        term_range = [max_val - fraction_range, max_val, upper, upper]
        feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
    else:
        term_range = [min_val + (i - 1) * fraction_range, min_val + i * fraction_range,
                      min_val + (i + 1) * fraction_range]
        feature_dict[var][term_name] = fuzz.trimf(feature_dict[var].universe, term_range)

# Fuzzy Output Variable

output_variable = rules_df.columns[-1]

var = output_variable
lower = fuzzy_lower[-1].astype(float)
min_val = fuzzy_min[-1].astype(float)
max_val = fuzzy_max[-1].astype(float)
upper = fuzzy_upper[-1].astype(float)
num_class = num_classes[-1].astype(float)

feature_dict[var] = {}
fraction_range = (max_val - min_val) / (num_class - 1)
universe = np.arange(lower, upper + 1, 1)
universe_dict[var] = universe
for i, term_name in enumerate(crispy_output):
    if i == 0:
        feature_dict[var] = ctrl.Consequent(universe, var)
        term_range = [lower, lower, min_val, min_val + fraction_range]
        feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
    elif i == num_class - 1:
        term_range = [max_val - fraction_range, max_val, upper, upper]
        feature_dict[var][term_name] = fuzz.trapmf(feature_dict[var].universe, term_range)
    else:
        term_range = [min_val + (i - 1) * fraction_range, min_val + i * fraction_range,
                      min_val + (i + 1) * fraction_range]
        feature_dict[var][term_name] = fuzz.trimf(feature_dict[var].universe, term_range)

# Fuzzy Rules

rule_dict = {}

n_rules = rules_df.shape[0]
n_features = rules_df.shape[1]

for i in range(n_rules):
    term_input = None
    term_output = None

    for j in range(n_features):

        if j == 0:
            term_input = feature_dict[rules_df.columns.tolist()[j]][rules_df.iloc[i, j]]

        elif j == n_features - 1:
            term_output = feature_dict[rules_df.columns.tolist()[j]][rules_df.iloc[i, j]]

        elif not pd.isna(rules_df.iloc[i, j]):
            term_input = term_input & feature_dict[rules_df.columns.tolist()[j]][rules_df.iloc[i, j]]

        else:
            continue

    rule_dict[i] = ctrl.Rule(antecedent=term_input, consequent=term_output)

# No Defuzzification Needed

# Control System Creation & Simulation & Saving

rule_vector = []
for i in range(len(rule_dict)):
    rule_vector.append(rule_dict[i])

dss_ctrl = ctrl.ControlSystem(rule_vector)
dss = ctrl.ControlSystemSimulation(dss_ctrl, flush_after_run=100 * 100 + 1)

dss_file_path = 'model_dss.pkl'
with open(dss_file_path, 'wb') as file:
    pickle.dump(dss, file)


# Decision System Application for Recommending the Next 3 Days

test_set, recommendations = multiple_recommendation(dss, data_dss, weather_set, xgb_model)


# Print Results

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(f"\nCurrent Date: {datetime.strptime(current_date, '%Y-%m-%d').strftime('%Y-%m-%d')}")
print("\n========== Test Set Data ==========")
print(f"\nThese are the Data for Sensor {sensor_id}\n")
print(test_set.round(2))

print("\n========== Output Results ==========")
print(f"\nThese are the Irrigation Recommendations based on Sensor {sensor_id}\n")
for date, result in recommendations:
    print(f"{date}: {result}")

print("\n========== Ground Truth ==========")
print(f"\nThese are the Real Irrigation that occurred according on the Nearest Sprinkler to Sensor {sensor_id}\n")
print(true_irrigation.round(2))
