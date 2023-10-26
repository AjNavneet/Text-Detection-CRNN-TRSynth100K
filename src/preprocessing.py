import os
import config
import pandas as pd

# Import utility function to save objects
from utils import save_obj

if __name__ == "__main__":
    # Load configuration settings from the "config" module
    image_path = config.image_path
    label_path = config.label_path
    char2int_path = config.char2int_path
    int2char_path = config.int2char_path
    data_file_path = config.data_file_path

    # Read labels from a text file
    labels = pd.read_table(label_path, header=None)
    # Fill missing label values with "null"
    labels.fillna("null", inplace=True)

    # Get a list of all image file names in the specified directory
    image_files = os.listdir(image_path)
    image_files.sort()
    # Create full paths for the images
    image_files = [os.path.join(image_path, i) for i in image_files]

    # Find the unique characters in the labels
    unique_chars = list({l for word in labels[0] for l in word})
    unique_chars.sort()
    # Create maps from character to integer and integer to character
    char2int = {a: i + 1 for i, a in enumerate(unique_chars)}
    int2char = {i + 1: a for i, a in enumerate(unique_chars)}

    # Save the character-to-integer and integer-to-character maps as objects
    save_obj(char2int, char2int_path)
    save_obj(int2char, int2char_path)

    # Create a data file containing image paths and corresponding labels
    data_file = pd.DataFrame({"images": image_files, "labels": labels[0]})
    data_file.to_csv(data_file_path, index=False)
