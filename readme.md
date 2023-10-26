# Text Detection using Convolutional Recurrent Neural Networks on TRSynth100K

## Business Objective

Text detection is the process of detecting text in images, and it has numerous applications, including solving CAPTCHAs, identifying vehicles by reading license plates, and more. This project combines Convolutional Neural Networks (CNNs) for image analysis with Recurrent Neural Networks (RNNs) for sequential data processing, creating a model known as Convolutional Recurrent Neural Network (CRNN). The focus of this model is on images with single-line text.

---

## Data Description

We will use the TRSynth100K dataset, which consists of approximately 100,000 images along with their labels in a text file.

- [TRSynth100K Dataset](https://www.kaggle.com/eabdul/textimageocr)

The dataset contains 100k images containing texts in them. Each image is of size 40x160 pixels. The text in the images are given as labels. The goal is to identify the text in the image.

---

## Aim

The objective of this project is to build a CRNN network capable of detecting and predicting the single-line text in a given image.

---

## Tech Stack

- Language: `Python`
- Libraries: `cv2`, `torch`, `numpy`, `pandas`, `albumentations`, `matplotlib`, `pickle`

---

## Approach

1. **Importing the Required Libraries**
2. **Download the Required Dataset**
3. **Pre-processing**
   - Find null values and replace them with the string "null"
   - Create a data frame with image paths and corresponding labels (save as a CSV file)
   - Create a mapping from characters to integers and save it in pickle format (char2int.pkl)
   - Create a mapping from integers to characters and save it in pickle format (int2char.pkl)
4. **Define Configurations and Paths**
5. **Training the Model**
   - Split the data into training and validation sets
   - Create training and validation datasets
   - Define the loss function
   - Create the CRNN model
   - Define the optimizer
   - Training loop
   - Save the trained model
6. **Predictions**
   - Select an image for prediction
   - Apply augmentations
   - Obtain model output
   - Apply softmax and extract label predictions
   - Convert integer predictions to a string using the character mapping
   - Display the output
   
---

## Modular Code Overview

1. **input**: Contains all the data for analysis, including two pickle files and one CSV file.
   - int2char.pkl
   - char2int.pkl
   - data_file.csv

2. **src**: The most important folder of the project, containing modularized code for all project steps.
   - **source** folder: Contains dataset.py and network.py files
   - Various other files for training, configuration, prediction, pre-processing, and utilities

1. **output**: Contains the best-fitted model trained for this data, which can be loaded and used for future tasks without retraining.

2. **lib**: A reference folder containing the original Jupyter notebook used in the project.

---

## Key Concepts Explored

1. Understanding the business problem
2. CRNN Architecture
3. Connectionist Temporal Classification (CTC) loss function
4. Making predictions with the model
5. Interpreting the generated results

---

## Getting Started

To install the dependencies run:
```
pip install -r requirements.txt
```

### Data processing
To process the data, run:
```
python preprocessing.py
```

### Train the model
To train the model, run:
```
python train.py
```

### Use the model to make predictions
To make predictions on an image, run:
```python predict.py ----test_img path_to_test_img```

---