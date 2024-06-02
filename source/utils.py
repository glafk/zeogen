import pickle
import os

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