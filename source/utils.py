import pickle
import os
import shutil
import json

import wandb
from wandb.apis import InternalApi
import torch
from omegaconf import OmegaConf

# Function to load the list from the pickle file
def load_objects(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            return pickle.load(file)
    else:
        return []

# Function to save the list to the pickle file
def save_objects(objects, pickle_file):
    with open(pickle_file, 'wb') as file:
        pickle.dump(objects, file)

# Function to add an object to the list and save it
def add_object(new_object, pickle_file):
    objects = load_objects(pickle_file)
    objects.append(new_object)
    save_objects(objects, pickle_file)


def load_from_wandb(experiment_name):
    api = wandb.Api()

    # Define the project and optionally the entity
    project = 'zeogen'
    entity = 'glafk'  # Optional, leave as None if not using

    # Get all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Search for the run by name
    matched_run = None
    for run in runs:
        if run.name == experiment_name:
            matched_run = run
            break

    if matched_run:
        run_id = matched_run.id
        print(f"Run ID for experiment '{experiment_name}' is: {run_id}")

    if run is None:
        raise ValueError(f"No run found with the name '{experiment_name}'")

    # List artifacts associated with the run
    artifacts = run.logged_artifacts()
    artifact_name = 'model:latest'

    # Retrieve the artifact
    artifact = api.artifact(f"{run.entity}/{run.project}/{artifact_name}")

    # Download the artifact
    artifact_dir = artifact.download()

    # Assuming the model file is named 'model.ckpt'
    model_path = os.path.join(artifact_dir, 'model.ckpt')

    model = torch.load_from_checkpoint(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Loaded model from run {run.experiment.name}.")

    wandb.finish()
    # Clean up downloaded files
    shutil.rmtree(artifact_dir)

    return model


def log_config_to_wandb(config, artifact_name="experiment_config", auxiliary_config=False):
    config = OmegaConf.to_container(config, resolve=True)
    if not auxiliary_config:
        wandb.config.update(config, allow_val_change=True)

    # Save the configuration to a file
    config_dir = "temp"
    os.makedirs(config_dir, exist_ok=True)
    config_filename = os.path.join(config_dir, "config.json")
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file)

    # Create an artifact
    artifact = wandb.Artifact(artifact_name, type='config')
    artifact.add_file(config_filename)

    # Log the artifact to wandb
    wandb.log_artifact(artifact)

    # Clean up the file so that it doesn't hang around
    os.remove(config_filename)


def retrieve_artifacts_by_name(experiment_name, artifact_type='dataset', project='zeogen', entity='glafk'):
    api = wandb.Api()
    
    # Retrieve the runs with matching experiment name
    runs = api.runs(f"{entity}/{project}", {"config.expname": experiment_name})

    artifact_files = []
    
    for run in runs:
        # Retrieve the artifacts associated with each run
        artifacts = run.use_artifacts(type=artifact_type)
        
        for artifact in artifacts:
            # Download the artifact files
            artifact_dir = artifact.download()
            artifact_files.extend([os.path.join(artifact_dir, file) for file in artifact.manifest.entries.keys()])
    
    return artifact_files