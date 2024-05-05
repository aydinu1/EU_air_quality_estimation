import os
import numpy as np
import argparse

import joblib
from comet_ml import Experiment, ExistingExperiment
from comet_ml import API

from dotenv import load_dotenv
import yaml
from attrdict2 import AttrDict
from utils.postgresql_utils import read_data_from_table, create_table, insert_df_to_table, delete_rows, run_vacuum
from utils.helper_functions import download_artifact, download_model, encode_city_name
import datetime

# Specify the path to config file
this_dir = os.path.dirname(__file__)
config_file = os.path.join(os.path.dirname(this_dir), 'config.yaml')
# Open and read the config file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))

# Specify path to comet-ml config file
comet_config_file = os.path.join(os.path.dirname(this_dir), 'comet_config.yaml')
with open(comet_config_file, 'r') as file:
    comet_config = AttrDict(yaml.safe_load(file))

# get API keys
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')
COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')

DB_USER_NAME = os.getenv('DB_USER_NAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_URL = os.getenv('DB_URL')

# set the seed
np.random.seed(config.seed)


def main():
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('-ex', '--create_new_experiment', action='store_true', help='Flag for model training')

    args = parser.parse_args()

    # Access the parsed arguments
    create_new_experiment = args.create_new_experiment
    # set up an experiment in cometml for logging the artifacts
    if create_new_experiment:
        experiment = Experiment(
            api_key=COMET_API_KEY,
            project_name="air_quality_estimation"
        )
    else:
        experiment = ExistingExperiment(api_key=COMET_API_KEY,
                                        previous_experiment=comet_config.experiment_id)

    api = API(api_key=COMET_API_KEY)

    # get encoder
    status = download_artifact(artifact_name=config.encoder_artifact_name,
                               experiment=experiment,
                               api=api,
                               workspace=COMET_WORKSPACE,
                               local_dirname=config.local_model_download_dirname,
                               version="latest")
    if status:
        print("Encoder downloaded successfully!")

    encoder = joblib.load(os.path.join(config.local_model_download_dirname, f"{config.encoder_artifact_name}.pkl"))

    # get model
    model = download_model(model_name=config.model_name,
                           api=api,
                           workspace=COMET_WORKSPACE,
                           local_dirname=config.local_model_download_dirname,
                           version="latest")

    # get inference data
    df_inference = get_inference_data(encoder=encoder)

    # make predictions
    df_prediction = make_prediction(df_inference, model)
    # save the inference data with predictions
    handle_prediction_data(df_prediction=df_prediction)

    # run vacuum to optimize DB
    run_vacuum(DB_URL, DB_USER_NAME, DB_PASSWORD)

    experiment.end()


def get_inference_data(encoder):
    start_date = (datetime.datetime.now()).strftime(format="%Y-%m-%d")
    end_date = (datetime.datetime.now() + datetime.timedelta(days=2)).strftime(format="%Y-%m-%d")

    query = f"SELECT * FROM {config.inference_data_table_name} " \
            f"WHERE time >= '{start_date}' " \
            f"AND time <= '{end_date}'"

    dfx = read_data_from_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                               table_name=config.inference_data_table_name,
                               query=query)

    dfx, _ = encode_city_name(dfx, label_encoder=encoder, save_encoder=False)

    return dfx


def make_prediction(df, model):
    df_city_names = df["city"]
    df_time = df["time"]

    # make predictions
    df = df.drop(["time", "city"], axis=1)
    X_inference = df.to_numpy()
    predictions = model.predict(X_inference)
    # predictions cannot be lower than 0
    predictions[predictions < 0] = 0

    # add the predictions, city and time
    df["predicted_pm25"] = predictions
    df["city"] = df_city_names
    df["time"] = df_time
    return df


def handle_prediction_data(df_prediction):
    # define data schema
    schema = """time TIMESTAMP,
                city VARCHAR(20),
                predicted_pm25 FLOAT,
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
                 table_name=config.prediction_data_table_name,
                 schema=schema)
    # delete older data from the table in case there is
    delete_rows(DB_URL,
                DB_USER_NAME,
                DB_PASSWORD,
                table_name=config.prediction_data_table_name,
                n_days=30)
    # insert the data
    df_prediction = df_prediction.drop(["city_encoded"], axis=1)
    insert_df_to_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                       config.prediction_data_table_name, df_prediction,
                       expected_columns=[*config.prediction_data_expected_columns])


if __name__ == "__main__":
    main()
