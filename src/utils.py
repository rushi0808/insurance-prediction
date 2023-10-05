import os
import pickle


def save_object(file_path, object):
    with open(file_path, "wb") as f:
        pickle.dump(object, f)


def load_object(file_path):
    with open(file_path, "rb") as f:
        object = pickle.load(f)
        return object
