import pickle
import os
import json

import wandb
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

def masked_bce_loss(predictions, targets, num_atoms, max_atoms):
    """
    Computes binary cross-entropy loss with a mask based on the number of atoms in each crystal.
    
    Args:
        predictions (torch.Tensor): Predicted probabilities of shape [batch_size, max_atoms].
        targets (torch.Tensor): Ground truth values of shape [num_atoms.sum()]. 
        num_atoms (torch.Tensor): Number of valid atoms in each crystal [batch_size].
        max_atoms (int): Maximum number of atoms in any crystal unit cell.

    Returns:
        torch.Tensor: Scalar masked loss value.
    """
    batch_size = predictions.size(0)
    
    # Create a mask based on the number of valid atoms
    mask = torch.arange(max_atoms).expand(batch_size, max_atoms).to(predictions.device) < num_atoms.unsqueeze(1)  # Shape: [batch_size, max_atoms]

    # Filter out valid predictions (flatten predictions and mask to match the flattened target size)
    valid_predictions = predictions[mask]  # Only take valid atoms for loss calculation
    valid_targets = targets  # Targets already have the valid atoms only

    # Compute binary cross-entropy loss on valid atoms
    loss = F.binary_cross_entropy(valid_predictions, valid_targets, reduction='none')

    return loss

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


def is_best_model(item):
    return item.type == "model" and "best" in item.aliases


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
    artifact_list = list(artifacts)

    best_model = next(filter(is_best_model, artifact_list))

    assert best_model is not None

    # Download the artifact
    model_dir = best_model.download()

    # Assuming the model file is named 'model.ckpt'
    model_path = os.path.join(model_dir, 'model.ckpt')

    return model_path, model_dir


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