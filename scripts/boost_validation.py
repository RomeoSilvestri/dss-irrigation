# Path & Libraries

import os
import sys
import warnings

sys.path.append(os.path.abspath('dss'))
warnings.filterwarnings('ignore')

import numpy as np
import random
import pandas as pd
from ctgan import CTGAN

random.seed(0)
np.random.seed(0)

file_path = '../data/val_data.xlsx'
val_data = pd.read_excel(file_path)
discrete_columns = [
    'Decision',
]

ctgan = CTGAN(verbose=True, epochs=200)
ctgan.fit(val_data, discrete_columns)

boost_val_data = ctgan.sample(1000)

input_variables = {
    'Last Avg Tensiometer': (50, 600),
    'Predicted Avg Tensiometer': (50, 600),
    'Predicted Rain Amount': (0, 40),
    'Predicted Max Temperature': (5, 40)
}

for column, (min_val, max_val) in input_variables.items():
    boost_val_data = boost_val_data[(boost_val_data[column] >= min_val) & (boost_val_data[column] <= max_val)]

boost_val_data.to_csv('../data/boost_val_data.csv', index=False)





def apply_decision_system(dec_sup_sys, input_dataset):
    output_array = []
    for index, row in input_dataset.iterrows():
        kwargs = row.to_dict()
        for key, value in kwargs.items():
            dec_sup_sys.input[key] = value
        dec_sup_sys.compute()
        output = dec_sup_sys.output['decision']
        output = fuzzy_decision(output)
        output_array.append(output)

    return output_array


dd = pd.read_csv('synthetic_data.csv').iloc[:200, :-1]
dd.columns = dd.columns.str.replace(' ', '_').str.lower()

output_results = apply_decision_system(dss, dd)


#print(output_results)


def calculate_prediction_error(predicted, actual):
    errors = 0
    total = len(predicted)
    for pred, act in zip(predicted, actual):
        if pred != act:
            errors += 1
    percentage_error = (errors / total) * 100
    return percentage_error


reference_vector = synthetic_data.iloc[:200, -1].tolist()
error_percentage = calculate_prediction_error(output_results, reference_vector)

print(output_results)
print(f"Error Percentage: {error_percentage:.2f}%")
