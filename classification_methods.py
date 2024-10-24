import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import importlib as il
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data.sampler import Sampler
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix
from mlcm import mlcm
import seaborn as sns

def read_and_combine_csvs(folder_path):
    all_data = []
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the CSV file with a different encoding
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Concatenate all data into a single DataFrame
    if all_data:  # Make sure there's data to concatenate
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        raise ValueError("No objects to concatenate")

    return combined_df

def read_csvs_by_group(folder_path):
    group_data = {}

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.startswith('assigned_classes_grp') and filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Extract group number from filename
            group_number = int(filename.split('_grp')[-1].split('.')[0])

            try:
                # Read the CSV file
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                group_data[group_number] = df
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Total groups read: {len(group_data)}")
    return group_data

# Custom dataset class to load images from subfolders and CSV
class CustomImageDatasetFromCSV(Dataset):
    def __init__(self, root_dir, combined_df, transform=None):
        self.root_dir = root_dir
        self.df = combined_df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        # Search for the image in subfolders
        img_path = None
        for subdir, _, files in os.walk(self.root_dir):
            if img_name in files:
                img_path = os.path.join(subdir, img_name)
                break

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any subfolder of {self.root_dir}")

        image = Image.open(img_path).convert("RGB")

        # Get labels (assuming the rest of the columns are the labels)
        labels = self.df.iloc[idx, 1:].values.astype('float')

        if self.transform:
            image = self.transform(image)

        return image, labels

def plot_histograms_for_groups(group_data):
    for group_number, df in group_data.items():
        # Sum the frequency of each label
        label_frequencies = df.iloc[:, 1:].sum()

        # Plot the histogram for the group
        plt.figure(figsize=(12, 6))
        label_frequencies.plot(kind='bar')
        plt.title(f'Frequency of Each Label for Group {group_number}')
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

def extract_label_frequencies(group_data):
    # Dictionary to store label frequencies for each label across all groups
    label_frequencies_by_group = {label: [] for label in group_data[1].columns[1:]}

    for group_number, df in group_data.items():
        label_frequencies = df.iloc[:, 1:].sum()
        for label, frequency in label_frequencies.items():
            label_frequencies_by_group[label].append(frequency)

    return label_frequencies_by_group

# refer to https://plotly.com/python/box-plots/
def plot_interactive_box_plot(label_frequencies_by_group):
    # Prepare data for the box plot
    data = []
    for label, frequencies in label_frequencies_by_group.items():
        for frequency in frequencies:
            data.append({'Label': label, 'Frequency': frequency})

    # Create the interactive box plot
    fig = px.box(pd.DataFrame(data), x='Label', y='Frequency', points="all")

    fig.update_layout(
        title="Distribution of Label Frequencies Across Groups",
        xaxis_title="Labels",
        yaxis_title="Frequency",
        hovermode="closest",
        template="plotly_white",
        height=800
    )

    # Show the plot
    fig.show()

# refer to https://discuss.pytorch.org/t/imbalanceddatasetsampler-dataloader-sampler/143996
class CustomImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        self.dataset = dataset
        self.indices = list(range(len(self.dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # Get label distribution from dataset
        label_counts = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label = tuple(label)  # Convert label to a tuple to use it as a key in the dictionary
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Calculate the weight for each sample based on class frequency
        weights = [1.0 / label_counts[tuple(self._get_label(dataset, idx))] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        # Access the labels from the DataFrame inside the dataset
        # changed for this specific case
        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.df.iloc[dataset.indices[idx], 1:].values
        else:
            return dataset.df.iloc[idx, 1:].values

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, len(self.indices), replacement=True))

    def __len__(self):
        return len(self.indices)

# Define the CNN model
class ChocolateCNN(nn.Module):
    def __init__(self):
        super(ChocolateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 25 * 6, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 9)  # 9 labels (for 9 classification categories)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(-1, 64 * 25 * 6)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer with 9 nodes for the 9 categories
        # x = torch.sigmoid(x)  # Sigmoid for multi-label classification
        return x


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.stem = nn.Conv2d(3, 32, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 9) # 9 labels (for 9 classification categories)


    def forward(self, x):
        x = self.bn1(F.relu(self.stem(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn2(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn3(F.relu(self.conv2(x)))
        # x = F.max_pool2d(x, 2, stride=2)
        x = F.max_pool2d(x, (2, 1), stride=(2, 1))
        x = self.dropout1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

def get_data_loaders(image_dir, input_size, combined_df):
        # Define transforms for your dataset (resize to match model input size)
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize images
        transforms.ToTensor(),
    ])

    dataset = CustomImageDatasetFromCSV(root_dir=image_dir, combined_df=combined_df, transform=transform)

    # Assuming `dataset` is an instance of your CustomImageDataset
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_indices = val_dataset.indices if isinstance(val_dataset, torch.utils.data.Subset) else list(range(len(val_dataset)))
    # Create the DataLoader for the training set with the CustomImbalancedDatasetSampler
    train_loader = DataLoader(
        train_dataset,
        sampler= CustomImbalancedDatasetSampler(train_dataset),  # Using the custom sampler for balanced batches
        batch_size= 32
    )

    # Create the DataLoader for the validation set without sampling
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader , val_indices

### Model 2: ResNet-50 ###
def get_resnet50_model():
    resnet50 = models.resnet50(weights= ResNet50_Weights.IMAGENET1K_V1)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, 9)  # Modify for multi-label classification
    # resnet50 = resnet50.to(device)

    return resnet50

# Training loop
def train_model1(device, model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    # Training and validation loops
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        hard_correct = 0
        hard_total = 0
        all_labels = []
        all_predictions = []
        misclassified_images = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

                hard_correct += (predicted == labels).all(dim=1).sum().item()
                hard_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Collect misclassified images
                for i in range(len(images)):
                    if not (predicted[i] == labels[i]).all():
                        misclassified_images.append({
                            'filename': val_loader.dataset.indices[i] if isinstance(val_loader.dataset, torch.utils.data.Subset) else i,
                            'label': labels[i].cpu().numpy(),
                            'predicted': predicted[i].cpu().numpy(),
                            'image': images[i].cpu()
                        })

        avg_val_loss = running_val_loss / len(val_loader)
        accuracy = correct / total * 100
        hard_accuracy = hard_correct / hard_total * 100

        val_losses.append(avg_val_loss)
        cm = confusion_matrix(np.array(all_labels).reshape(-1), np.array(all_predictions).reshape(-1))

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%, Hard Accuracy: {hard_accuracy:.2f}%")

    return model, optimizer, avg_train_loss, avg_val_loss, accuracy, hard_accuracy, train_losses, val_losses, cm, misclassified_images

def save_model(model, optimizer, num_epochs, avg_train_loss, avg_val_loss, accuracy, hard_accuracy, train_losses, val_losses, cm, val_indices,misclassified_images, save_path):
    # Determine the model type based on the model's class name
    model_type = model.__class__.__name__  # This will get the class name like 'ChocolateCNN', 'ImprovedCNN', or 'ResNet50'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': model_type,  # Store the determined model type dynamically
        'epoch': num_epochs,
        'loss_function': 'BCEWithLogitsLoss',
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'Accuracy': accuracy,
        'Hard_Accuracy': hard_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'confusion_matrix': cm,
        'val_indices': val_indices,
        'misclassified_images': misclassified_images
    }, save_path)

    print(f"Model saved successfully as {model_type} at {save_path}")


def load_model_with_metadata1(model_path, device, weights_only=None):
    checkpoint = torch.load(model_path, map_location=device, weights_only=weights_only)
    model_type = checkpoint['model_type']
    print(model_type)
    print(model_type)
    # Dynamically determine the model class
    if model_type == 'ChocolateCNN':
        model_class = ChocolateCNN
        input_size = (100, 25)
    elif model_type == 'ImprovedCNN':
        model_class = ImprovedCNN
        input_size = (100, 25)
    elif model_type == 'ResNet50':
        model_class = get_resnet50_model
        input_size = (224, 224)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = model_class().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    metadata = {
        'epoch': checkpoint['epoch'],
        'loss_function': checkpoint['loss_function'],
        'model_type': model_type,
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'Accuracy': checkpoint['Accuracy'],
        'Hard_Accuracy': checkpoint['Hard_Accuracy'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'confusion_matrix': checkpoint['confusion_matrix'],
        'val_indices': checkpoint['val_indices'],
        'misclassified_images': checkpoint['misclassified_images']
    }

    print(f"Loaded model type: {model_type}, epoch: {metadata['epoch']}, validation accuracy: {metadata['Accuracy']}%")
    return model, optimizer, metadata , input_size

def plot_losses(train_losses, val_losses, title="Training and Validation Loss Over Epochs"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_confusion_matrix(cm, class_names):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     cax = ax.matshow(cm, cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix', pad=20)
#     fig.colorbar(cax)

#     ax.set_xticks(np.arange(len(class_names)))
#     ax.set_yticks(np.arange(len(class_names)))

#     ax.set_xticklabels(class_names, rotation=45, ha="right")
#     ax.set_yticklabels(class_names)

#     plt.xlabel('Predicted')
#     plt.ylabel('True')

#     for (i, j), val in np.ndenumerate(cm):
#         plt.text(j, i, f'{val}', ha='center', va='center', color='red')

#     plt.tight_layout()
#     plt.show()

def plot_multi_label_confusion_matrix(labels, predictions, class_names):
    # Initialize a confusion matrix for each class
    num_classes = labels.shape[1]
    cm_per_class = []

    # Calculate confusion matrix for each label
    for i in range(num_classes):
        cm = confusion_matrix(labels[:, i], predictions[:, i])
        cm_per_class.append(cm)

    # Plot the confusion matrices for each class
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Adjust for a 3x3 grid if there are 9 classes
    axes = axes.flatten()

    for i, cm in enumerate(cm_per_class):
        ax = axes[i]
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Confusion Matrix for {class_names[i]}', pad=20)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])

        # Annotate the confusion matrix values
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f'{val}', ha='center', va='center', color='red')

    plt.tight_layout()
    plt.show()

def plot_mlcm(matrix_to_plot, label_names):
    # Create a heatmap with seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_to_plot, annot=True, fmt='g', cmap='Blues', xticklabels=label_names, yticklabels=label_names)

    # Set labels, title, and axis ticks
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()

def show_misclassified_images(misclassified_images, num_images=20):
    rows = (num_images + 3) // 4  # Calculate the number of rows needed
    plt.figure(figsize=(20, rows * 6))  # Adjust figure size accordingly

    for i, info in enumerate(misclassified_images[:num_images]):
        plt.subplot(rows, 4, i + 1)  # Arrange in a grid of 4 columns
        image = info['image'].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

        plt.imshow(image)
        plt.title(f"File: {os.path.basename(info['filename'])}\n"
                  f"True: {info['label']}, Pred: {info['predicted']}\n"
                  f"Score: {info['score']}", fontsize=10)  # Adjusted for smaller font size
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#   Make predictions on all images in a folder and save them as a CSV file.
def make_predictions(model, device, folder_path, transform, labels_names, output_csv='predictions.csv'):

    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    model.to(device)
    model.eval()  # Set model to evaluation mode

    for image_file in image_files:
        # Load and preprocess the image
        img_path = os.path.join(folder_path, image_file)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            predicted_probs = torch.sigmoid(outputs)  # Apply sigmoid for multi-label classification
            # Format tensor output to 4 decimal places
            torch.set_printoptions(sci_mode=False, precision=4)

            print(predicted_probs)
            predicted_labels = (predicted_probs > 0.5).int()  # Convert probabilities to binary predictions

        predicted_labels = predicted_labels.cpu().numpy().flatten()

        # Store the results
        predictions.append({'filename': image_file, **dict(zip(labels_names, predicted_labels))})

        # Prepare the annotated text for the current image
        annotated_text = ", ".join([label for label, value in zip(labels_names, predicted_labels) if value == 1])

        # Display the image with its annotated text
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
            break

    # Close the OpenCV window
    cv2.destroyAllWindows()

    # Convert the predictions to a DataFrame and save as CSV with proper encoding for German letters
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {output_csv}")

    return df






# def make_predictions(model, device, folder_path, transform, labels_names, output_csv='predictions.csv'):
#     """
#     Make predictions on all images in a folder and save them as a CSV file.

#     Parameters:
#         model (nn.Module): The loaded model for making predictions.
#         device (torch.device): Device to perform inference on.
#         folder_path (str): Path to the folder containing images.
#         transform (torchvision.transforms): Transformations to apply to the images.
#         labels_names (list): List of label names.
#         output_csv (str): Path to save the predictions as a CSV file.

#     Returns:
#         pd.DataFrame: DataFrame containing the filenames and their predictions.
#     """
#     # Get list of image files in the folder
#     image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     predictions = []

#     model.to(device)
#     model.eval()  # Set model to evaluation mode

#     for image_file in image_files:
#         # Load and preprocess the image
#         img_path = os.path.join(folder_path, image_file)
#         image = Image.open(img_path).convert('RGB')
#         image = transform(image)
#         image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

#         # Make prediction
#         with torch.no_grad():
#             outputs = model(image)
#             predicted_probs = torch.sigmoid(outputs)  # Apply sigmoid for multi-label classification
#             predicted_labels = (predicted_probs > 0.5).int()  # Convert probabilities to binary predictions

#         predicted_labels = predicted_labels.cpu().numpy().flatten()

#         # Store the results
#         predictions.append({'filename': image_file, **dict(zip(labels_names, predicted_labels))})

#     # Convert the predictions to a DataFrame and save as CSV with proper encoding for German letters
#     df = pd.DataFrame(predictions)
#     df.to_csv(output_csv, index=False, encoding='utf-8-sig')
#     print(f"Predictions saved to {output_csv}")

#     # Display each image with predictions
#     for image_file in image_files:
#         img_path = os.path.join(folder_path, image_file)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (100, 400))  # Resize back to original size if needed

#         # Annotate the image with the predicted labels
#         annotated_text = ", ".join([label for label, value in zip(labels_names, predicted_labels) if value == 1])
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.5
#         thickness = 1
#         position = (10, 30)  # Adjusted for the smaller image size

#         # Annotate with the predicted labels
#         cv2.putText(img, f'Predicted: {annotated_text}', position, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

#         # Display the image
#         cv2.imshow('Predictions', img)

#         # Wait until the user presses a key
#         key = cv2.waitKey(0)  # Wait indefinitely for a key press
#         if key == ord('q'):  # Press 'q' to quit the display
#             print("Display canceled by user.")
#             break

#     # Close the OpenCV window
#     cv2.destroyAllWindows()

#     return df
