import pickle

def save_obj(obj, path):
    """
    Save a Python object to a file using pickle.

    :param obj: The object to be saved
    :param path: The path to the file where the object will be saved
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    """
    Load a Python object from a file using pickle.

    :param path: The path to the file from which the object will be loaded
    :return: The loaded Python object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
