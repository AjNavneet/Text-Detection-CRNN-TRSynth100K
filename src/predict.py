# Import necessary libraries and modules
import cv2
import torch
import albumentations
from utils import load_obj
from source.network import ConvRNN
from argparse import ArgumentParser

if __name__ == "__main__":
    # Define command-line arguments for testing
    parser = ArgumentParser()
    parser.add_argument("--test_img", default="data/TRSynth100K/images/00000017.jpg", help="path to test image")
    parser.add_argument("--model_path", default="models/model.pth", help="path to the saved model")
    parser.add_argument("--int2char_path", default="data/int2char.pkl", help="path to int2char")
    opt = parser.parse_args()

    # Load the integer to character mapping dictionary
    int2char = load_obj(opt.int2char_path)
    # Determine the number of classes
    n_classes = len(int2char)

    # Create the ConvRNN model
    model = ConvRNN(n_classes)
    # Load the model weights from the specified path
    model.load_state_dict(torch.load(opt.model_path))
    # Port the model to CUDA if a GPU is available
    if torch.cuda.is available():
        model.cuda()
    # Set the model to evaluation mode
    model.eval()

    # Check if CUDA is available, and define mean and std for image normalization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Load and preprocess the test image
    img = cv2.imread(opt.test_img)
    img_aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )
    augmented = img_aug(image=img)
    img = augmented["image"]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    # Create a batch dimension (batch of a single image)
    img = torch.unsqueeze(img, 0)
    # Move the image array to CUDA if available
    img = img.to(device)

    # Pass the image through the model
    out = model(img)
    # Remove the batch dimension
    out = torch.squeeze(out, 0)
    # Apply softmax to the predictions
    out = out.softmax(1)
    # Find the character with the highest probability for each timeframe
    pred = torch.argmax(out, 1)
    # Convert the prediction tensor to a list
    pred = pred.tolist()
    # Use 'ph' for the special character
    int2char[0] = "ph"
    # Convert integer predictions to character labels
    out = [int2char[i] for i in pred]

    # Collapse the output by removing repeated characters and 'ph'
    res = list()
    res.append(out[0])
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            res.append(out[i])
    res = [i for i in res if i != "ph"]
    # Join the characters to form the recognized text
    res = "".join(res)
    # Print the recognized text
    print(res)
