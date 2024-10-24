import os
import sys
import importlib as il
import glob

import pandas as pd
import numpy as np
import importlib as il
import matplotlib.pyplot as plt

import tqdm
from PIL import Image
import cv2


import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torchvision import transforms, models
from mlcm import mlcm



CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PARENT_DIR)

import classification_methods as clm


pjoin = os.path.join


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = pjoin(PARENT_DIR, "labels")
combined_df = clm.read_and_combine_csvs(folder_path)

# print(combined_df.head())
group_data = clm.read_csvs_by_group(folder_path)

# Class names
label_names = [
    "leichte durchscheinende Füllung",
    "mittlere durchscheinende Füllung",
    "starke durchscheinende Füllung",
    "Bläschen",
    "Füllung über Rand hinaus",
    "Kratzer/Späne auf Deckel",
    "Zelle falsch erkannt",
    "small bright area on upper border",
    "sonstiger Fehler"
]

class Manager:

    def __inti__(self):
        pass
        self.metadata: dict = None
        self.model_path = None



    def load_model(self, model_path):
        """
        :param model_path:  example: classification_models/CNN_3.pth
        """

        self.model_path = model_path


        # Load the model, optimizer, and metadata
        # model, optimizer, metadata = clm.load_model_with_metadata(model_path, device)
        self.model, self.optimizer, self.metadata, self.input_size = clm.load_model_with_metadata1(model_path, device, weights_only=False)
        # Access metadata
        num_epochs = self.metadata['epoch']
        model_type = self.metadata['model_type']
        train_loss = self.metadata['train_loss']
        val_loss = self.metadata['val_loss']
        accuracy = self.metadata['Accuracy']
        hard_accuracy = self.metadata['Hard_Accuracy']
        # hard_accuracy = metadata['Hard_Accuracy']

        # Reconstruct the validation DataLoader using val_indices
        val_indices = self.metadata['val_indices']
        self.transform = transforms.Compose([
                transforms.Resize(self.input_size),  # Resize images
                transforms.ToTensor(),
            ])
        self.loaded_dataset = clm.CustomImageDatasetFromCSV(root_dir='./results', combined_df=combined_df, transform=self.transform)
        self.val_dataset = torch.utils.data.Subset(self.loaded_dataset, val_indices)
        val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)


    def plot_training_curves(self):
        # Plot training and validation losses
        clm.plot_losses(self.metadata['train_losses'], self.metadata['val_losses'])


    def make_predictions(self, input_image_dir_path=None, limit=None, display=False):

        if input_image_dir_path is None:
            input_image_dir_path = pjoin(PARENT_DIR, "predict_inputs")

        self.input_image_dir_path = input_image_dir_path

        # following code is based on clm.make_predictions


        # Get list of image files in the folder
        self.img_fnames_for_prediction = [f for f in os.listdir(self.input_image_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.predictions = []
        self.outputs = []
        torch.set_printoptions(sci_mode=False, precision=4)

        self.model = self.model

        self.model.to(device)
        self.model.eval()  # Set model to evaluation mode

        for img_fname in tqdm.tqdm(self.img_fnames_for_prediction):
            # Load and preprocess the image

            img_fpath = pjoin(self.input_image_dir_path, img_fname)
            image = Image.open(img_fpath).convert('RGB')
            image = self.transform(image)
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

            # Make prediction
            with torch.no_grad():
                image_outputs = self.model(image)
                self.outputs.append(image_outputs)
                predicted_probs = torch.sigmoid(image_outputs)  # Apply sigmoid for multi-label classification
                # Format tensor output to 4 decimal places

                # print(predicted_probs)
                predicted_labels = (predicted_probs > 0.5).int()  # Convert probabilities to binary predictions

            predicted_labels = [int(elt) for elt in predicted_labels.cpu().numpy().flatten()]

            # Store the results
            self.predictions.append({'filename': img_fname, **dict(zip(label_names, predicted_labels))})

            # Prepare the annotated text for the current image
            annotated_text = ", ".join([label for label, value in zip(label_names, predicted_labels) if value == 1])

            # Display the image with its annotated text

            if display:
                res = self.display_prediction(img_fpath, annotated_text)
                if res == "break":
                    break


    def display_prediction(self, img_path, annotated_text):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 400))  # Resize to (width=100, height=400)

        # Create a wider canvas with space for the text below the image
        canvas_width = 400
        canvas_height = img.shape[0] + 100
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background

        # Center the image on the canvas
        x_offset = (canvas_width - img.shape[1]) // 2
        canvas[:img.shape[0], x_offset:x_offset + img.shape[1]] = img

        # Annotate with the predicted labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(annotated_text, font, font_scale, thickness)[0]
        text_x = (canvas_width - text_size[0]) // 2  # Center the text horizontally
        text_y = img.shape[0] + 60  # Position the text 60 pixels below the image

        cv2.putText(canvas, annotated_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Predictions', canvas)

        # Wait until the user presses a key
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == ord('q'):  # Press 'q' to quit the display
            print("Display canceled by user.")
            return "break"
