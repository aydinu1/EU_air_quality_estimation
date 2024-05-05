import uuid
import numpy as np
from sklearn.preprocessing import LabelEncoder
from comet_ml import Artifact
import joblib
import os
import yaml
from attrdict2 import AttrDict


# Specify the path to config file
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(os.path.dirname(parent_dir), 'config.yaml')
# Open and read the YAML file
with open(config_file, 'r') as file:
    config = AttrDict(yaml.safe_load(file))


def encode_city_name(df, label_encoder=None, save_encoder=True):
    # encode city name
    if label_encoder is None:
        label_encoder = LabelEncoder()
    # Fit the encoder and transform the data
    encoded = label_encoder.fit_transform(df["city"])
    # set the encoded city column
    # df = df.drop(columns=["city"], axis=1)
    df["city_encoded"] = encoded

    if save_encoder:
        joblib.dump(label_encoder, os.path.join(os.path.dirname(parent_dir), config.local_models_dir,
                                                f"{config.encoder_artifact_name}.pkl"))

    return df, label_encoder


def download_artifact(experiment, artifact_name, api, workspace, local_dirname, version="latest"):
    # get the artifact information
    info = api.get_artifact_details(artifact_name=artifact_name, workspace=workspace)
    versions = np.sort([x["version"] for x in info["versions"]])
    if version.casefold() != "latest":
        if version not in versions:
            print(f"Given version {version} is not available for the artifact {artifact_name}."
                  f"Available versions are {versions}. Latest version will be downloaded now.")
            version = "latest"

    # download training data from artifacts
    if not os.path.exists(local_dirname):
        os.mkdir(local_dirname)
    logged_artifact = experiment.get_artifact(artifact_name, version_or_alias=version)
    local_artifact = logged_artifact.download(local_dirname, overwrite_strategy="OVERWRITE")
    return True


def download_model(api, workspace, model_name, local_dirname, version="latest"):
    model = api.get_model(workspace=workspace, model_name=model_name)
    versions = model.find_versions()
    if version.casefold() == "latest":
        model.get_details(versions[0])
        version = versions[0]

    if not os.path.exists(local_dirname):
        os.mkdir(local_dirname)
    # Download the registered model:
    model.download(version=version,
                   output_folder=local_dirname,
                   expand=True)

    retrieved_model = joblib.load(os.path.join(local_dirname, f"{model_name}.pkl"))
    print("Model downloaded successfully!")
    return retrieved_model


def log_encoder(encoder, artifact_name, experiment):
    # save the label encoder locally
    if not os.path.exists("models"):
        os.mkdir("models")
    encoder_local_path = os.path.join("models", f"{artifact_name}.pkl")
    joblib.dump(encoder, encoder_local_path)
    # log the label encoder as artifact
    artifact = Artifact(name=artifact_name, artifact_type="labelencoder")
    artifact.add(encoder_local_path)
    experiment.log_artifact(artifact)


def create_experiment_id(comet_config_file='comet_config.yaml'):
    with open(comet_config_file, 'w') as file:
        experiment_id = uuid.uuid4().hex
        yaml.dump({'experiment_id': experiment_id}, file)
    return experiment_id


def get_model_metadata(model, model_version="latest"):
    versions = model.find_versions()
    if model_version.casefold() == "latest":
        version = versions[0]
    model_assets = model.get_assets(version)
    model_metadata = model_assets[-1]["metadata"]
    return model_metadata


def log_model(model, model_name, experiment, model_save_local_path, model_metadata=None):
    model_save_local_path = os.path.join(model_save_local_path, f"{model_name}.pkl")
    print(f"Model is being saved to local path:{model_save_local_path}")
    joblib.dump(model, model_save_local_path)

    # log the model
    experiment.log_model(model_name, model_save_local_path,
                         metadata=model_metadata)
    return "Model is registered successfully"


def log_metrics(metrics: dict, experiment):
    # log the metrics
    experiment.log_metrics(metrics)
    return "Model metrics logged successfully"

