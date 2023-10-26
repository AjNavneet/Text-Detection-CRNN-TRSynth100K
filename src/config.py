# Number of training epochs
epochs = 100

# Number of samples in each batch during training
batch_size = 32

# Path to save the trained model
model_path = "models/model.pth"

# Path to the pickled dictionary for character-to-integer mapping
char2int_path = "data/char2int.pkl"

# Path to the pickled dictionary for integer-to-character mapping
int2char_path = "data/int2char.pkl"

# Path to the data file (CSV, perhaps containing training data)
data_file_path = "data/data_file.csv"

# Path to the directory containing images
image_path = "data/TRSynth100K/images"

# Path to the file containing labels (text labels corresponding to images)
label_path = "data/TRSynth100K/labels.txt"
