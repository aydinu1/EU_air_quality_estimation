import datetime

import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from comet_ml.integration.sklearn import log_model
from comet_ml import Experiment, ExistingExperiment, API

from utils.postgresql_utils import read_data_from_table, run_vacuum
from utils.helper_functions import encode_city_name, log_encoder, create_experiment_id, log_metrics, download_model, \
    log_model
import os
import argparse
from dotenv import load_dotenv
import yaml
from attrdict2 import AttrDict

# Specify the path to config file
this_dir = os.path.dirname(__file__)
config_file = os.path.join(os.path.dirname(this_dir), 'config.yaml')
# Open and read the config file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))

# get API keys
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')
COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')

DB_USER_NAME = os.getenv('DB_USER_NAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_URL = os.getenv('DB_URL')

# set the seed
SEED = 42
np.random.seed(config.seed)


def main():
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('-cv', '--cross_validate', action='store_true', help='Flag for model training')
    parser.add_argument('-ex', '--create_new_experiment', action='store_true', help='Flag for model training')

    args = parser.parse_args()

    # Access the parsed arguments
    cross_validate = args.cross_validate
    create_new_experiment = args.create_new_experiment

    # set up an experiment in cometml for logging the artifacts
    if create_new_experiment:
        experiment_id = create_experiment_id()
        experiment = Experiment(
            api_key=COMET_API_KEY,
            project_name="air_quality_estimation",
            experiment_key=experiment_id
        )
    else:
        # Specify path to comet-ml config file
        comet_config_file = os.path.join(os.path.dirname(this_dir), 'comet_config.yaml')
        with open(comet_config_file, 'r') as file:
            comet_config = AttrDict(yaml.safe_load(file))
        experiment = ExistingExperiment(api_key=COMET_API_KEY,
                                        previous_experiment=comet_config.experiment_id)

    api = API(api_key=COMET_API_KEY)

    # get the train and test data
    X_train, X_test, y_train, y_test = get_train_test_data(experiment)

    # train_model model
    challenger_model, model_metadata = train_model(X_train=X_train, X_test=X_test,
                                                   y_train=y_train, y_test=y_test,
                                                   do_CV=cross_validate)

    # compare the challenger to existing model
    replace_model = compare_models(api=api,
                                   X_test=X_test, y_test=y_test,
                                   mae_test_challenger=model_metadata["MAE_test"],
                                   existing_model_ver="latest")

    if replace_model:
        print("Challenger model is accepted")
        # save the new model
        save_model_locally(challenger_model,
                           model_save_path=config.local_models_dir,
                           model_name=config.model_name)
        # log the model
        log_model(model=challenger_model,
                  model_name=config.model_name,
                  experiment=experiment,
                  model_metadata=model_metadata,
                  model_save_local_path=config.local_models_dir)
        # log the metrics
        log_metrics(metrics=model_metadata, experiment=experiment)
        # register the model
        experiment.register_model(config.model_name)

    # run vacuum to optimize DB
    run_vacuum(DB_URL, DB_USER_NAME, DB_PASSWORD)
    experiment.end()


def get_train_test_data(experiment):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime(format="%Y-%m-%d")
    end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(format="%Y-%m-%d")

    query = f"SELECT * FROM {config.historical_data_table_name} " \
            f"WHERE time >= '{start_date}' " \
            f"AND time <= '{end_date}'"

    dfx = read_data_from_table(DB_URL, DB_USER_NAME, DB_PASSWORD,
                               table_name=config.historical_data_table_name,
                               query=query)

    # set time as index
    dfx = dfx.set_index(["time"])
    dfx.index.name = "time"

    # encode city name
    dfx, label_encoder = encode_city_name(dfx)
    # log the encoder
    log_encoder(label_encoder,
                artifact_name=config.encoder_artifact_name,
                experiment=experiment)

    # get the target
    dfy = dfx.pop(config.target_column)

    # dfx, label_encoder = encode_city_name(dfx)
    # drop the city column before training, since we have it encoded
    if "city_encoded" in dfx.columns:
        dfx = dfx.drop(columns=["city"], axis=1)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(dfx, dfy,
                                                        test_size=config.test_size,
                                                        random_state=config.seed)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, do_CV=False):
    model = XGBRegressor(**config.model_hyper_params)
    model_metadata = {}
    if do_CV:
        cv = RepeatedKFold(n_splits=config.n_splits,
                           n_repeats=config.n_repeats,
                           random_state=config.seed)

        scores = cross_val_score(model,
                                 X_train,
                                 y_train,
                                 scoring='neg_mean_absolute_error',
                                 cv=cv,
                                 n_jobs=-1)

        scores = np.absolute(scores)
        print(f'Mean of MAE training scores: {scores.mean()}, STD of MAE training scores: {scores.std()}')
        model_metadata["Mean_of_MAE_training_scores"] = scores.mean()
        model_metadata["STD_of_MAE_training_scores"] = scores.std()

    # fit the model on whole train set
    model.fit(X_train, y_train)

    # evaluate
    mae_train = evaluate_model(model, X_train, y_train)
    mae_test = evaluate_model(model, X_test, y_test)
    model_metadata["MAE_train"] = mae_train
    model_metadata["MAE_test"] = mae_test
    print("MAE on train dataset:", mae_train)
    print("MAE on test dataset:", mae_test)

    return model, model_metadata


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    # evaluate
    mae = mean_absolute_error(y, y_pred)
    return mae


def save_model_locally(model, model_save_path, model_name):
    # Save the trained XGBoost model using joblib
    model_save_path = os.path.join(model_save_path, f"{model_name}.pkl")
    joblib.dump(model, model_save_path)
    print(f"Model {model_name} saved locally to path {model_save_path}.")


def compare_models(api, X_test, y_test, mae_test_challenger, existing_model_ver="latest"):
    print("Comparing models")
    # check if there is any existing models
    existing_models = api.get_registry_model_names(workspace=COMET_WORKSPACE)
    # if there are no models, return True so that we can register the challenger model
    if not any(existing_models):
        return True
    # get existing model
    existing_model = download_model(model_name=config.model_name,
                                    api=api,
                                    workspace=COMET_WORKSPACE,
                                    local_dirname=config.local_model_download_dirname,
                                    version=existing_model_ver)

    mae_test_existing = evaluate_model(existing_model, X_test, y_test)

    print("MAE_test existing model:", mae_test_existing)
    print("MAE_test challenger model:", mae_test_challenger)

    if mae_test_challenger < mae_test_existing:
        return True
    return False


if __name__ == "__main__":
    main()



