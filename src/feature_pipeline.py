import os
import datetime
import numpy as np
import argparse
from dotenv import load_dotenv

from utils.postgresql_utils import insert_df_to_table, create_table, delete_rows, run_vacuum
from utils.get_data import get_historical_data, get_forecast_data
import yaml
from attrdict2 import AttrDict

# Specify the path to config file
this_dir = os.path.dirname(__file__)
config_file = os.path.join(os.path.dirname(this_dir), 'config.yaml')
# Open and read the YAML file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))

# Database credentials
load_dotenv()
DB_USER_NAME = os.getenv('DB_USER_NAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_URL = os.getenv('DB_URL')

# set seed
np.random.seed(config.seed)


def main():
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('-t', '--get_training_data_arg', action='store_true',
                        help='Flag for getting the training data')
    parser.add_argument('-i', '--get_inference_data_arg', action='store_true',
                        help='Flag for getting the training data')

    args = parser.parse_args()

    # Access the parsed arguments
    get_training_data_arg = args.get_training_data_arg
    get_inference_data_arg = args.get_inference_data_arg

    if get_training_data_arg:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime(format="%Y-%m-%d")
        end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(format="%Y-%m-%d")
        status = get_train_data(start_date=start_date, end_date=end_date)
        if status:
            print("Training data fetched and logged successfully.")

    if get_inference_data_arg:
        status, _ = get_inference_data()
        if status:
            print("Inference data fetched and logged successfully.")

    # run vacuum to optimize DB
    run_vacuum(DB_URL, DB_USER_NAME, DB_PASSWORD)


def get_train_data(store_training_data=True, start_date=None, end_date=None):
    df_historical = get_historical_data(start_date=start_date, end_date=end_date)
    # TODO: data validation with pydantic here

    # feature engineering steps should be applied here if any needed
    df_preprocessed = pre_process_data(df_historical)

    if store_training_data:
        # define data schema
        schema = """time TIMESTAMP,
                    pm2_5 FLOAT,
                    city VARCHAR(20),
                    latitude FLOAT,
                    longitude FLOAT,
                    temperature_2m FLOAT,
                    relativehumidity_2m INTEGER,
                    precipitation FLOAT,
                    cloudcover INTEGER,
                    cloudcover_low INTEGER,
                    cloudcover_mid INTEGER,
                    cloudcover_high INTEGER,
                    windspeed_10m FLOAT,
                    winddirection_10m INTEGER,
                    windgusts_10m FLOAT,
                    month INTEGER,
                    hour INTEGER,
                    PRIMARY KEY (time, city)"""
        # create table if it doesn't exist
        create_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                     table_name=config.historical_data_table_name,
                     schema=schema)
        # delete older data from the table in case there is
        delete_rows(DB_URL,
                    DB_USER_NAME,
                    DB_PASSWORD,
                    table_name=config.historical_data_table_name,
                    n_days=60)

        # insert the data
        insert_df_to_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                           config.historical_data_table_name, df_preprocessed,
                           expected_columns=[*config.train_data_expected_columns])
    return True


def get_inference_data(store_inference_data=True):
    df_forecast = get_forecast_data()
    # TODO: data validation with pydantic here

    df_preprocessed = pre_process_data(df_forecast)

    # feature engineering steps should be applied here if any needed

    if store_inference_data:
        # define data schema
        schema = """time TIMESTAMP,
                    city VARCHAR(20),
                    latitude FLOAT,
                    longitude FLOAT,
                    temperature_2m FLOAT,
                    relativehumidity_2m INTEGER,
                    precipitation FLOAT,
                    cloudcover INTEGER,
                    cloudcover_low INTEGER,
                    cloudcover_mid INTEGER,
                    cloudcover_high INTEGER,
                    windspeed_10m FLOAT,
                    winddirection_10m INTEGER,
                    windgusts_10m FLOAT,
                    month INTEGER,
                    hour INTEGER,
                    PRIMARY KEY (time, city)"""
        # create table if it doesn't exist
        create_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                     table_name=config.inference_data_table_name,
                     schema=schema)
        # delete older data from the table in case there is
        delete_rows(DB_URL,
                    DB_USER_NAME,
                    DB_PASSWORD,
                    table_name=config.inference_data_table_name,
                    n_days=30)
        # insert the data
        insert_df_to_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                           config.inference_data_table_name, df_preprocessed,
                           expected_columns=[*config.inference_data_expected_columns])
    return True, df_preprocessed


def pre_process_data(df):
    # drop nans
    df = df.dropna(axis=0).copy()
    # get month and hour of the date
    df["month"] = df["time"].dt.month.astype('int')
    df["hour"] = df["time"].dt.hour.astype('int')
    return df


if __name__ == "__main__":
    main()
