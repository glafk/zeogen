import pickle
import os
import shutil

import wandb
from wandb.apis import InternalApi
import torch

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
    print(f"Added object: {new_object}")


def load_from_wandb(model_name):
    wandb.init(project="zeogen", entity="glafk")
    api = InternalApi()
    runs = api.runs("your_entity_name", project="your_project_name")  # Specify your entity and project names
    for run in runs:
        if run.experiment.name == model_name:
            artifact = wandb.use_artifact(f'glafk/zeogen/{run.id}:latest', type='model')

            artifact_dir = artifact.download()
            model_path = os.path.join(artifact_dir, 'model.ckpt')
            model = torch.load_from_checkpoint(model_path)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            print(f"Loaded model from run {run.experiment.name}.")

            wandb.finish()
            # Clean up downloaded files
            shutil.rmtree(artifact_dir)

            return model
