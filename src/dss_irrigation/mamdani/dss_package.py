# Libraries

#from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from ipywidgets import widgets, Output
from IPython.display import display
import kaleido
import numpy as np
import matplotlib.pyplot as plt
from openmeteo_requests import Client
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests_cache
from retry_requests import retry
# from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


# Irrigation Value Converters

def categorize_irrigation(decision_defuzzy):
    turn_number = decision_defuzzy / 650.0
    if turn_number <= 0.25:
        return 'Not Recommended'
    elif 0.25 < turn_number <= 0.75:
        return 'Half Turn'
    elif 0.75 < turn_number <= 1.5:
        return 'Single Turn'
    else:
        return 'Double Turn'


def reverse_categorize_irrigation(category):
    if category == 'Not Recommended':
        return 0.0
    elif category == 'Half Turn':
        return 325.0
    elif category == 'Single Turn':
        return 650.0
    else:
        return 1300.0


def fuzzy_decision(decision_defuzzy):
    if decision_defuzzy <= 0.5:
        return 'Not Recommended'
    elif 0.5 < decision_defuzzy <= 1.5:
        return 'Half Turn'
    elif 1.5 < decision_defuzzy <= 2.5:
        return 'Single Turn'
    else:
        return 'Double Turn'


# Select & Load Data

def select_data_ml(file_path, sensor_id, current_date, days_predicted=3):
    df = pd.read_csv(file_path)

    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = (current_date + timedelta(days=1))

    df = df[(df['sensor_id'] == sensor_id) & (df['date'] == start_date.strftime('%Y-%m-%d'))]
    df.set_index('date', inplace=True)
    df.drop(columns=['sensor_id', 'avg_tens'], inplace=True)

    new_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_predicted)]
    new_rows = pd.DataFrame(index=new_dates, columns=df.columns).fillna(np.nan)
    df = pd.concat([df, new_rows])
    df[['max_temp', 'sum_rain', 'sum_irr']] = np.nan, np.nan, 0.0

    lag_columns = {f"lag{i}_columns": [col for col in df.columns if col.endswith(f'lag{i}')] for i in
                   range(1, days_predicted)}

    for i in range(1, days_predicted):
        for col in lag_columns[f'lag{i}_columns']:
            if i == 1:
                df.loc[df.index[i], col.replace(f'lag{i}', f'lag{i + 1}')] = df.loc[df.index[0], col]
                df.loc[df.index[i + 1], col.replace(f'lag{i}', f'lag{i + 2}')] = df.loc[df.index[0], col]
            else:
                df.loc[df.index[i - 1], col.replace(f'lag{i}', f'lag{i + 1}')] = df.loc[df.index[0], col]

    df = df.astype('float64')
    df.index = df.index.astype(str)
    df = df.sort_index(axis=1)

    return df


def load_weather_data(latitude, longitude, current_date, days_predicted=3):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = Client(session=retry_session)

    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (current_date + timedelta(days=days_predicted + 1)).strftime('%Y-%m-%d')

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    # url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": ["temperature_2m_max", "rain_sum"],
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }
    response = openmeteo.weather_api(url, params=params)[0]

    daily = response.Daily()
    predicted_max_temperature = daily.Variables(0).ValuesAsNumpy()
    predicted_rain_amount = daily.Variables(1).ValuesAsNumpy()

    weather_set = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ), "max_temp": predicted_max_temperature, "sum_rain": predicted_rain_amount}

    weather_set = pd.DataFrame(data=weather_set)[1:].reset_index(drop=True)
    weather_set['date'] = weather_set['date'].dt.date
    weather_set.set_index('date', inplace=True)

    weather_set = weather_set.astype('float64')
    weather_set.index = weather_set.index.astype(str)

    return weather_set


def combine_data_dss(df, weather_set):
    days = df.shape[0]

    for i in range(1, days):
        weather_set[f'max_temp_lag{i}'] = weather_set['max_temp'].shift(i)
        weather_set[f'sum_rain_lag{i}'] = weather_set['sum_rain'].shift(i)

    test_data = df.combine_first(weather_set)

    return test_data


def load_true_irrigation(file_path, sensor_id, current_date, days_predicted=3):
    df = pd.read_csv(file_path)

    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    start_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (current_date + timedelta(days=days_predicted + 1)).strftime('%Y-%m-%d')

    df = df[(df['sensor_id'] == sensor_id) & (df['date'] >= start_date) & (df['date'] <= end_date)]
    df.drop(columns=['sensor_id', 'avg_tens'], inplace=True)
    df.set_index('date', inplace=True)
    df.index.name = None

    true_irrigation = pd.DataFrame({
        'AVG Tensiometer': df['avg_tens_lag1'].shift(-1).dropna().round(2),
        'MAX Temperature': df['max_temp_lag1'].shift(-1).dropna().round(2),
        'SUM Rain': df['sum_rain_lag1'].shift(-1).dropna().round(2),
        'Water Quantity': df['sum_irr_lag1'].shift(-1).dropna().round(2),
        'Irrigation': df['sum_irr_lag1'].shift(-1).dropna().apply(categorize_irrigation)
    })

    return true_irrigation


def load_scenarios(dataframe, date_value, sensor_id_value):
    filtered_data = dataframe[(dataframe['date'] == date_value) & (dataframe['sensor_id'] == sensor_id_value)]
    filtered_data = filtered_data.drop(columns=['date', 'sensor_id'])

    num_columns = len(filtered_data.columns)
    num_rows = num_columns // 4
    reshaped_data = filtered_data.to_numpy().reshape(num_rows, 4, -1)

    columns_names = ['Day_1', 'Day_2', 'Day_3', 'Day_4']
    rows_names = ['Tensiometer', 'Rain', 'Hum_Avg', 'Temp_Max', 'Temp_Min', 'Temp_Avg']
    df = pd.DataFrame(reshaped_data.reshape(-1, 4), columns=columns_names, index=rows_names)
    df = df.astype(int)

    return df


# Data Visualization

def plot_decision_surface_tens(dss, predicted_rain_amount, predicted_max_temperature, save_path=None):
    upsampled = np.linspace(300, 500, 50)
    x, y = np.meshgrid(upsampled, upsampled)
    z = np.zeros_like(x)

    result_cache = {}

    for i in range(50):
        for j in range(50):
            input_key = (x[i, j], y[i, j], predicted_rain_amount, predicted_max_temperature)

            if input_key in result_cache:
                z[i, j] = result_cache[input_key]
            else:
                dss.input['last_avg_tensiometer'] = x[i, j]
                dss.input['predicted_avg_tensiometer'] = y[i, j]
                dss.input['predicted_rain_amount'] = predicted_rain_amount
                dss.input['predicted_max_temperature'] = predicted_max_temperature
                dss.compute()
                z[i, j] = dss.output['decision']
                result_cache[input_key] = z[i, j]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

    ax.view_init(30, 200)
    ax.set_xlabel('Last Avg Tensiometer')
    ax.set_ylabel('Predicted Avg Tensiometer')
    ax.set_zlabel('Decision')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")

    plt.show()


def plot_decision_surface_temp(dss, predicted_rain_amount, last_avg_tensiometer, save_path=None):
    upsampled_tensiometer = np.linspace(300, 500, 50)
    upsampled_temperature = np.linspace(20, 40, 50)
    x, y = np.meshgrid(upsampled_tensiometer, upsampled_temperature)
    z = np.zeros_like(x)

    result_cache = {}

    for i in range(50):
        for j in range(50):
            input_key = (last_avg_tensiometer, x[i, j], predicted_rain_amount, y[i, j])

            if input_key in result_cache:
                z[i, j] = result_cache[input_key]
            else:
                dss.input['last_avg_tensiometer'] = last_avg_tensiometer
                dss.input['predicted_avg_tensiometer'] = x[i, j]
                dss.input['predicted_rain_amount'] = predicted_rain_amount
                dss.input['predicted_max_temperature'] = y[i, j]
                dss.compute()
                z[i, j] = dss.output['decision']
                result_cache[input_key] = z[i, j]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=300, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=20, cmap='viridis', alpha=0.5)

    ax.view_init(30, 200)
    ax.set_xlabel('Predicted Avg Tensiometer')
    ax.set_ylabel('Predicted Max Temperature')
    ax.set_zlabel('Decision')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")

    plt.show()


def plot_decision_surface_rain(dss, predicted_max_temperature, last_avg_tensiometer, save_path=None):
    upsampled_rain = np.linspace(0, 25, 50)
    upsampled_tensiometer = np.linspace(300, 500, 50)
    x, y = np.meshgrid(upsampled_rain, upsampled_tensiometer)
    z = np.zeros_like(x)

    result_cache = {}

    for i in range(50):
        for j in range(50):
            input_key = (last_avg_tensiometer, y[i, j], x[i, j], predicted_max_temperature)

            if input_key in result_cache:
                z[i, j] = result_cache[input_key]
            else:
                dss.input['last_avg_tensiometer'] = last_avg_tensiometer
                dss.input['predicted_avg_tensiometer'] = y[i, j]
                dss.input['predicted_rain_amount'] = x[i, j]
                dss.input['predicted_max_temperature'] = predicted_max_temperature
                dss.compute()
                z[i, j] = dss.output['decision']
                result_cache[input_key] = z[i, j]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=0, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=300, cmap='viridis', alpha=0.5)

    ax.view_init(30, 200)
    ax.set_xlabel('Predicted Rain Amount')
    ax.set_ylabel('Predicted Avg Tensiometer')
    ax.set_zlabel('Decision')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")

    plt.show()


def create_plot_interface(dss):
    widgets_dict = {}
    parameter_names = ['predicted_rain_amount', 'predicted_max_temperature']

    for param in parameter_names:
        widgets_dict[param] = widgets.FloatText(description='')

    output_area = widgets.Output()

    def on_compute_button_clicked(button):
        with output_area:
            output_area.clear_output(wait=True)
            # Salva il grafico durante la visualizzazione
            save_path = 'decision_surface_plot.png'
            plot_decision_surface_tens(
                dss,
                widgets_dict['predicted_rain_amount'].value,
                widgets_dict['predicted_max_temperature'].value,
                save_path=save_path
            )

    compute_button = widgets.Button(description='Compute')
    compute_button.on_click(on_compute_button_clicked)

    input_boxes = [widgets.VBox([widgets.Label(f"{param.replace('_', ' ').title()}: "), widget])
                   for param, widget in widgets_dict.items()]

    display(*input_boxes, compute_button, output_area)


def compute_output(dss, rules_df, **kwargs):
    for key, value in kwargs.items():
        dss.input[key] = value
    dss.compute()

    decision_defuzzy = dss.output[rules_df.columns[-1]]
    fuzzy_result = fuzzy_decision(decision_defuzzy)
    return fuzzy_result


def create_decision_interface(dss, rules_df, n_parameters):
    widgets_dict = {}
    parameter_vector = []
    for i in range(n_parameters):
        parameter_name = rules_df.columns[i].replace('_', ' ').title()
        parameter_vector.append(parameter_name)
        widgets_dict[parameter_name] = widgets.FloatText(description='')

    input_boxes = [widgets.VBox([widgets.Label(f"{param}:"), widget])
                   for param, widget in zip(parameter_vector, widgets_dict.values())]

    output_area = Output()

    def on_compute_button_clicked(button):
        with output_area:
            output_area.clear_output()
            kwargs = {key: widget.value for key, widget in zip(rules_df.columns[:-1], widgets_dict.values())}
            fuzzy_result = compute_output(dss, rules_df, **kwargs)
            print(fuzzy_result)

    compute_button = widgets.Button(description='Compute')
    compute_button.on_click(on_compute_button_clicked)

    display(*input_boxes, compute_button, output_area)


def show_figure(sensor_target_id, soil1, soil2, irrigation1, irrigation2, rain, output_file):
    date_range = [datetime(2023, 5, 1) + timedelta(days=i) for i in range(len(soil1))]

    fig = make_subplots(rows=2, cols=1, subplot_titles=('<b>Observed</b>', '<b>Recommended</b>'),
                        vertical_spacing=0.1, shared_xaxes=True,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    fig.update_annotations(yshift=10)

    max_value = max(np.max(irrigation1), np.max(irrigation2), 1350)

    scaled_irrigation1 = np.clip(irrigation1, 0, max_value)
    scaled_irrigation1 = (scaled_irrigation1) / max_value

    scaled_irrigation2 = np.clip(irrigation2, 0, max_value)
    scaled_irrigation2 = (scaled_irrigation2) / max_value

    max_value = max(np.max(rain), 15)
    scaled_rain = np.clip(rain, 0, max_value)
    scaled_rain = (scaled_rain) / max_value

    # Soil Moisture (Observed)
    fig.add_trace(
        go.Scatter(x=date_range, y=np.round(soil1, 2),
                   mode='lines',
                   line=dict(color='black'),
                   marker_size=3,
                   marker_symbol='circle',
                   name='Soil Moisture',
                   showlegend=True,
                   marker_color='white',
                   marker=dict(line=dict(color='blue', width=2))
                   ),
        row=1, col=1, secondary_y=False
    )

    fig.add_shape(type="line", x0=date_range[0], x1=date_range[-1], y0=400, y1=400,
                  line=dict(color="red", width=2), row=1, col=1, secondary_y=False,
                  name="Upper Threshold", showlegend=True)
    fig.add_shape(type="line", x0=date_range[0], x1=date_range[-1], y0=200, y1=200,
                  line=dict(color="blue", width=2), row=1, col=1, secondary_y=False,
                  name="Lower Threshold", showlegend=True)

    # Irrigation (Observed), ora scalato tra 0 e 1
    fig.add_trace(
        go.Bar(
            x=date_range,
            y=scaled_irrigation1,
            name='Irrigation',
            marker_color='cyan',
            opacity=0.7,
            customdata=np.round(irrigation1, 2),
            hovertemplate='Irrigation: %{customdata}<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )

    # Rain (Observed), ora scalato tra 0 e 1
    fig.add_trace(
        go.Bar(
            x=date_range,
            y=scaled_rain,
            name='Rain',
            marker_color='green',
            opacity=0.7,
            customdata=np.round(rain, 2),
            hovertemplate='Rain: %{customdata}<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )

    # Soil Moisture (Recommended)
    fig.add_trace(
        go.Scatter(x=date_range, y=np.round(soil2, 2),
                   mode='lines',
                   line=dict(color='black'),
                   marker_size=3,
                   marker_symbol='circle',
                   name='Soil Moisture',
                   showlegend=False,
                   marker_color='white',
                   marker=dict(line=dict(color='blue', width=2))
                   ),
        row=2, col=1, secondary_y=False
    )

    fig.add_shape(type="line", x0=date_range[0], x1=date_range[-1], y0=400, y1=400,
                  line=dict(color="red", width=2), row=2, col=1, secondary_y=False)
    fig.add_shape(type="line", x0=date_range[0], x1=date_range[-1], y0=200, y1=200,
                  line=dict(color="blue", width=2), row=2, col=1, secondary_y=False)

    # Irrigation (Recommended), ora scalato tra 0 e 1
    fig.add_trace(
        go.Bar(
            x=date_range,
            y=scaled_irrigation2,
            name='Irrigation',
            marker_color='cyan',
            opacity=0.7,
            customdata=np.round(irrigation2, 2),
            hovertemplate='Irrigation: %{customdata}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1, secondary_y=True
    )

    # Rain (Recommended), ora scalato tra 0 e 1
    fig.add_trace(
        go.Bar(
            x=date_range,
            y=scaled_rain,
            name='Rain',
            marker_color='green',
            opacity=0.7,
            customdata=np.round(rain, 2),
            hovertemplate='Rain: %{customdata}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1, secondary_y=True
    )

    # Layout settings
    fig.update_layout(
        autosize=False,
        width=2000,
        height=1000,
        margin=dict(l=50, r=50, b=50, t=20, pad=4),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=26, family="Times New Roman")
        ),
        font=dict(size=26, family="Times New Roman")
    )

    fig.update_annotations(
        font=dict(size=28, family="Times New Roman")
    )

    max_soil = max(np.max(soil1), np.max(soil2), 700)
    max_irrigation = max(np.max(irrigation1), np.max(irrigation2), 1000)

    fig.update_xaxes(tickfont=dict(size=26, family="Times New Roman"), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=26, family="Times New Roman"), row=2, col=1)

    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, max_soil], title_text="Soil Moisture (mbar)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, max_irrigation], showgrid=False, title_text="Irrigation (Liters)", secondary_y=True,
                     row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, max_soil], title_text="Soil Moisture (mbar)", secondary_y=False, row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, max_irrigation], showgrid=False, title_text="Irrigation (Liters)", secondary_y=True,
                     row=2, col=1)

    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, 1], showgrid=False, title_text="Normalized Water Supply", secondary_y=True, row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=26, family="Times New Roman"), title_font=dict(size=26, family="Times New Roman"), range=[0, 1], showgrid=False, title_text="Normalized Water Supply", secondary_y=True, row=2, col=1)

    #fig.write_image(output_file, scale = 3)
    fig.show()


def show_figure2(sensor_target_id, soil1, soil2, soil3, irrigation1, irrigation2, irrigation3, rain, output_file):
    from datetime import datetime, timedelta
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    date_range = [datetime(2023, 5, 1) + timedelta(days=i) for i in range(len(soil1))]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=('<b>Observed</b>', '<b>Mamdani DSS Simulation</b>', '<b>Takagi-Sugeno DSS Simulation</b>'),
        vertical_spacing=0.1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    fig.update_annotations(yshift=10)

    # Normalizzazione irrigazione
    max_irrig = max(np.max(irrigation1), np.max(irrigation2), np.max(irrigation3), 1350)
    scaled_irrigation1 = np.clip(irrigation1, 0, max_irrig) / max_irrig
    scaled_irrigation2 = np.clip(irrigation2, 0, max_irrig) / max_irrig
    scaled_irrigation3 = np.clip(irrigation3, 0, max_irrig) / max_irrig

    # Normalizzazione pioggia
    max_rain = max(np.max(rain), 15)
    scaled_rain = np.clip(rain, 0, max_rain) / max_rain

    # === 1. Observed ===
    fig.add_trace(go.Scatter(x=date_range, y=np.round(soil1, 2),
                             mode='lines', line=dict(color='black'),
                             name='Soil Moisture', marker=dict(line=dict(color='blue', width=2))),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Bar(x=date_range, y=scaled_irrigation1, name='Irrigation',
                         marker_color='cyan', opacity=0.7,
                         customdata=np.round(irrigation1, 2),
                         hovertemplate='Irrigation: %{customdata}<extra></extra>'),
                  row=1, col=1, secondary_y=True)

    fig.add_trace(go.Bar(x=date_range, y=scaled_rain, name='Rain',
                         marker_color='green', opacity=0.7,
                         customdata=np.round(rain, 2),
                         hovertemplate='Rain: %{customdata}<extra></extra>'),
                  row=1, col=1, secondary_y=True)

    # === 2. Recommended ===
    fig.add_trace(go.Scatter(x=date_range, y=np.round(soil2, 2),
                             mode='lines', line=dict(color='black'),
                             name='Soil Moisture', showlegend=False,
                             marker=dict(line=dict(color='blue', width=2))),
                  row=2, col=1, secondary_y=False)

    fig.add_trace(go.Bar(x=date_range, y=scaled_irrigation2, name='Irrigation',
                         marker_color='cyan', opacity=0.7,
                         customdata=np.round(irrigation2, 2),
                         hovertemplate='Irrigation: %{customdata}<extra></extra>',
                         showlegend=False),
                  row=2, col=1, secondary_y=True)

    fig.add_trace(go.Bar(x=date_range, y=scaled_rain, name='Rain',
                         marker_color='green', opacity=0.7,
                         customdata=np.round(rain, 2),
                         hovertemplate='Rain: %{customdata}<extra></extra>',
                         showlegend=False),
                  row=2, col=1, secondary_y=True)

    # === 3. Alternative ===
    fig.add_trace(go.Scatter(x=date_range, y=np.round(soil3, 2),
                             mode='lines', line=dict(color='black'),
                             name='Soil Moisture', showlegend=False,
                             marker=dict(line=dict(color='blue', width=2))),
                  row=3, col=1, secondary_y=False)

    fig.add_trace(go.Bar(x=date_range, y=scaled_irrigation3, name='Irrigation',
                         marker_color='cyan', opacity=0.7,
                         customdata=np.round(irrigation3, 2),
                         hovertemplate='Irrigation: %{customdata}<extra></extra>',
                         showlegend=False),
                  row=3, col=1, secondary_y=True)

    fig.add_trace(go.Bar(x=date_range, y=scaled_rain, name='Rain',
                         marker_color='green', opacity=0.7,
                         customdata=np.round(rain, 2),
                         hovertemplate='Rain: %{customdata}<extra></extra>',
                         showlegend=False),
                  row=3, col=1, secondary_y=True)

    # === Soglie e stile ===
    for row in [1, 2, 3]:
        show_legend = (row == 1)  # Solo nel primo subplot
        fig.add_trace(
            go.Scatter(
                x=[date_range[0], date_range[-1]],
                y=[400, 400],
                mode='lines',
                line=dict(color='red', width=2),
                name='Dry Threshold',
                showlegend=show_legend
            ),
            row=row, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=[date_range[0], date_range[-1]],
                y=[200, 200],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Wet Threshold',
                showlegend=show_legend
            ),
            row=row, col=1, secondary_y=False
        )

    # === Layout ===
    max_soil = max(np.max(soil1), np.max(soil2), np.max(soil3), 700)

    fig.update_layout(
        autosize=False,
        width=2000,
        height=1500,
        margin=dict(l=50, r=50, b=50, t=20, pad=4),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="right", x=1, font=dict(size=26, family="Times New Roman")
        ),
        font=dict(size=26, family="Times New Roman")
    )

    fig.update_annotations(font=dict(size=28, family="Times New Roman"))

    for row in [1, 2, 3]:
        fig.update_xaxes(tickfont=dict(size=26, family="Times New Roman"), row=row, col=1)
        fig.update_yaxes(range=[0, max_soil], title_text="Soil Moisture (mbar)",
                         tickfont=dict(size=26, family="Times New Roman"),
                         title_font=dict(size=26, family="Times New Roman"),
                         secondary_y=False, row=row, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Normalized Water Supply",
                         showgrid=False,
                         tickfont=dict(size=26, family="Times New Roman"),
                         title_font=dict(size=26, family="Times New Roman"),
                         secondary_y=True, row=row, col=1)

    fig.write_image(output_file, scale=2)
    fig.show()


def apply_decision_system(dec_sup_sys, input_dataset):
    output_array = []

    for index, row in input_dataset.iterrows():
        kwargs = row.to_dict()
        for key, value in kwargs.items():
            dec_sup_sys.input[key] = value
        dec_sup_sys.compute()
        output = dec_sup_sys.output['decision']
        output = fuzzy_decision(output)
        output_array.append((row.name, output))

    return output_array


def multiple_recommendation(dss, df, weather_set, model):
    test_set = pd.DataFrame(columns=['last_avg_tensiometer', 'predicted_avg_tensiometer',
                                     'predicted_max_temperature', 'predicted_rain_amount'])

    output_results = []

    last_values = {'sum_irrigation': [df['sum_irr_lag1'][0], df['sum_irr_lag2'][0], df['sum_irr_lag3'][0]],
                   'avg_tensiometer': [df['avg_tens_lag1'][0], df['avg_tens_lag2'][0], df['avg_tens_lag3'][0]]}

    for day in range(len(df)):
        df_day = df.iloc[[day]].copy()

        if day != 0:
            for i in range(1, 4):
                df_day[f'sum_irr_lag{i}'] = last_values['sum_irrigation'][i - 1]
                df_day[f'avg_tens_lag{i}'] = last_values['avg_tensiometer'][i - 1]

        last_avg_tensiometer = last_values['avg_tensiometer'][0]
        predicted_avg_tensiometer = model.predict(df_day)[0]
        predicted_max_temperature = weather_set['max_temp'].iloc[day]

        if day + 2 < len(df):
            predicted_rain_amount = weather_set['sum_rain'].iloc[day] + weather_set['sum_rain'].iloc[day + 1] + weather_set['sum_rain'].iloc[day + 2]
        elif day + 1 < len(weather_set):
            predicted_rain_amount = weather_set['sum_rain'].iloc[day] + weather_set['sum_rain'].iloc[day + 1]
        else:
            predicted_rain_amount = weather_set['sum_rain'].iloc[day]

        # predicted_rain_amount = weather_set['sum_rain'].iloc[day]

        new_row = {
            'last_avg_tensiometer': last_avg_tensiometer,
            'predicted_avg_tensiometer': predicted_avg_tensiometer,
            'predicted_max_temperature': predicted_max_temperature,
            'predicted_rain_amount': predicted_rain_amount
        }
        test_set.loc[df_day.index[0]] = new_row

        result = apply_decision_system(dss, test_set.iloc[day:])[0]
        output_results.append(result)

        if result[1] != 'Not Recommended':
            df_day['sum_irr'] = reverse_categorize_irrigation(result[1])
            last_values['avg_tensiometer'][0] = model.predict(df_day)[0]
        else:
            last_values['avg_tensiometer'][0] = predicted_avg_tensiometer

        last_values['sum_irrigation'][0] = df_day['sum_irr'][0]
        last_values['sum_irrigation'][1] = df_day['sum_irr_lag1'][0]
        last_values['sum_irrigation'][2] = df_day['sum_irr_lag2'][0]
        last_values['avg_tensiometer'][1] = df_day['avg_tens_lag1'][0]
        last_values['avg_tensiometer'][2] = df_day['avg_tens_lag2'][0]

    test_set.columns = ['LAST AVG Tensiometer', 'PRED AVG Tensiometer', 'PRED MAX Temperature', 'PRED SUM Rain']
    test_set = test_set.astype('float64')

    return test_set, output_results


# Error Calculation

def compute_error(dss, validation_data, input_columns, output_column, error_metric='mae'):
    actual_outputs = validation_data[output_column].values
    predicted_outputs = []

    for i in range(len(validation_data)):
        for col in input_columns:
            dss.input[col] = validation_data.iloc[i][col]

        dss.compute()
        predicted_output = dss.output[output_column]
        predicted_outputs.append(predicted_output)

    predicted_outputs = np.array(predicted_outputs)
    distances = predicted_outputs - actual_outputs

    if error_metric == 'mae':
        error = np.mean(np.abs(distances))
    elif error_metric == 'rmse':
        error = np.sqrt(np.mean(distances ** 2))
    else:
        raise ValueError("Invalid Metric. Choose 'mae' or 'rmse'.")

    return error
