import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import numpy as np
import os

from dotenv import load_dotenv
import yaml
from attrdict2 import AttrDict

from src.utils.plotting import plot_air_quality_map
from src.utils.postgresql_utils import read_data_from_table


# Specify the path to config file
config_file = 'config.yaml'
# Open and read the YAML file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))

# get API keys
load_dotenv()
DB_USER_NAME = os.getenv('DB_USER_NAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_URL = os.getenv('DB_URL')

# set seed
np.random.seed(config.seed)


def get_output_data(start_date, end_date):
    query = f"SELECT * FROM {config.prediction_data_table_name} " \
            f"WHERE time >= '{start_date}' " \
            f"AND time <= '{end_date}'"

    df_plotting = read_data_from_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                                       table_name=config.prediction_data_table_name,
                                       query=query)

    return df_plotting


# Define GMT+0 timezone
timzezone = datetime.timezone(datetime.timedelta(hours=0))
start_date = (datetime.datetime.now(timzezone)).strftime(format="%Y-%m-%d")
end_date = (datetime.datetime.now(timzezone) + datetime.timedelta(days=2)).strftime(format="%Y-%m-%d")
df_output = get_output_data(start_date=start_date, end_date=end_date)

# drop duplicates
df_output = df_output.reset_index()
df_output = df_output.drop_duplicates(subset=['time', 'city']).set_index(["time"])
# get next hour for map
df_next_hour = df_output.loc[(df_output.index.day == datetime.datetime.now(timzezone).day) &
                             (df_output.index.hour == datetime.datetime.now(timzezone).hour+1)]
# get next 24 hours for the line chart
df_next_24hour = df_output.loc[(pd.to_datetime(df_output.index).tz_localize(timzezone) >=
                                datetime.datetime.now(timzezone)) &
                               (pd.to_datetime(df_output.index).tz_localize(timzezone) <=
                                datetime.datetime.now(timzezone) +
                                datetime.timedelta(hours=24))]
# title of the app
st.title("Prediction of Fine Particle Pollution in EU Capitals for the Next 24h (GMT+0)")

# create a Plotly figure
fig = plot_air_quality_map(df_next_hour, plot_prediction=True)

# display the Plotly figure
st.plotly_chart(fig)

# You can also add other content to your Streamlit app, such as text, widgets, or additional plots.
# For example, you can add a text section using st.markdown:
# st.markdown("Predictions by XGBoost.")

selected_city = st.selectbox("Select a city to plot predictions for the next 24h (GMT+0)",
                             df_next_24hour['city'].unique())

# Create a line plot for air pollution data based on the selected city
line_plot_fig = go.Figure()

# Filter the DataFrame based on the selected city
selected_city_data = df_next_24hour[df_next_24hour['city'] == selected_city].sort_index()

# Add the line plot to the figure
line_plot_fig.add_trace(
    go.Scatter(
        x=selected_city_data.index,  # Use index as x-axis (or specify a time-related column)
        y=selected_city_data['predicted_pm25'],
        mode='lines+markers',
        name='Air Pollution',
    )
)

# Customize the line plot (optional)
line_plot_fig.update_layout(
    title=f'Air Pollution in {selected_city}',
    xaxis_title='Time',  # Adjust as needed
    yaxis_title='Air Pollution',
)

# Display the line plot
st.plotly_chart(line_plot_fig)

# Adding a footer using st.markdown()
st.markdown(
    """
    <hr style="border: 1px solid #d3d3d3;">
    <p style='text-align: center; font-size:0.85em;'>Made by Ugur Aydin</p>
    """,
    unsafe_allow_html=True
)
