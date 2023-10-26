import torch
import config
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from utils import load_obj
from source.network import ConvRNN
from source.dataset import TRSynthDataset
from sklearn.model_selection import train_test_split

def train(model, dataloader, criterion, device, optimizer=None, test=False):
    """
    Function to train the model
    :param network: Model object
    :param loader: data loader
    :param loss_fn: loss function
    :param dvc: device (cpu or cuda)
    :param opt: optimizer
    :param test: True for validation (gradients won't be updated)
    :return: Average loss for the epoch
    """

    # Set mode to train or validation
    if test:
        model.eval()
    else:
        model.train()
    loss = []
    for inp, tgt, tgt_len in tqdm(dataloader):
        # Move tensors to the specified device
        inp = inp.to(device)
        tgt = tgt.to device)
        tgt_len = tgt_len.to(device)
        # Forward pass
        out = model(inp)
        out = out.permute(1, 0, 2)
        # Calculate input lengths for the data points
        # All have equal length of 40 since all images in
        # the dataset are of equal length
        inp_len = torch.LongTensor([40] * out.shape[1])
        # Calculate CTC Loss
        log_probs = nn.functional.log_softmax(out, 2)
        loss_ = criterion(log_probs, tgt, inp_len, tgt_len)

        if not test:
            # Update weights during training
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        loss.append(loss_.item())

    return np.mean(loss)

if __name__ == "__main__":
    data_file_path = config.data_file_path
    char2int_path = config.char2int_path
    epochs = config.epochs
    batch_size = config.batch_size
    model_path = config.model_path

    # Read the data file
    data_file = pd.read_csv(data_file_path)
    data_file.fillna("null", inplace=True)

    # Load character-to-integer mapping dictionary
    char2int = load_obj(char2int_path)

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is available() else "cpu")

    # Split the data into train and validation sets
    train_file, valid_file = train_test_split(data_file, test_size=0.2)

    # Create train and validation datasets
    train_dataset = TRSynthDataset(train_file, char2int)
    valid_dataset = TRSynthDataset(valid_file, char2int)

    # Define the loss function
    criterion = nn.CTCLoss(reduction="sum")
    criterion.to(device)

    # Number of classes
    n_classes = len(char2int)
    # Create a model object
    model = ConvRNN(n_classes)
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)

    # Define train and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)

    # Training loop
    best_loss = 1e7
    for i in range(epochs):
        print(f"Epoch {i + 1} of {epochs}...")
        # Run the train function for training
        train_loss = train(model, train_loader, criterion, device, optimizer, test=False)
        # Run the train function for validation
        valid_loss = train(model, valid_loader, criterion, device, test=True)
        print(f"Train Loss: {round(train_loss, 4)}, Valid Loss: {round(valid_loss, 4)}")
        if valid_loss < best_loss:
            print("Validation Loss improved, saving Model File...")
            # Save the model object
            torch.save(model.state_dict(), model_path)
            best_loss = valid_loss
