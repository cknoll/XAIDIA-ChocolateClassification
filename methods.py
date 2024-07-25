import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column, gridplot
from bokeh.transform import dodge
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ipydex import Container
from torch.utils.data.sampler import Sampler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_score_from_filename(filename):
    match = re.search(r's(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

def get_label_from_filename(filename,classification):
    # match = re.search(r's(\d+)_', filename)
    # if match:
    # score = int(match.group(1))
    score= get_score_from_filename(filename)
    if classification[0] <= score < classification[1]:
        return 0
    elif classification[1] <= score < classification[2]:
        return 1
    elif classification[2] <= score < classification[3]:
        return 2
        
    elif classification[3] <= score < classification[4]:
        return 3
    else:
        return 4

# Custom dataset class for our images
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, classification=[]):
        self.root_dir = root_dir
        self.transform = transform
        self.classification = classification
        self.image_files = []
        self.labels = []
        self.scores = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    try:
                        self.image_files.append(os.path.join(subdir, file))
                        self.labels.append(get_label_from_filename(file,self.classification))
                        self.scores.append(get_score_from_filename(file))
                    except ValueError as e:
                        print(e)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        img_path = self.image_files[idx]
        label = self.labels[idx]
        score = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, score , img_path

    def print_labels(self, num_samples=5):
        for i in range(min(num_samples, len(self.image_files))):
            print(f"Filename: {self.image_files[i]}, Label: {self.labels[i]}")
    
    def get_labels(self):
        return self.labels

class CustomImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        self.dataset = dataset
        self.indices = list(range(len(self.dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # Get label distribution
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(self.dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # Weight for each sample
        weights = [1.0 / label_to_count[self._get_label(self.dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        # Check if dataset is a Subset
        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.labels[dataset.indices[idx]]
        return dataset.labels[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def training_loop(model, num_epochs, criterion, optimizer, train_loader, val_loader, device):
    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels, scores, img_path in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        misclassified_images = []

        with torch.no_grad():
            for images, labels, scores, img_path in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # print(outputs)
                # print(outputs.shape)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            
                # Collect information on misclassified images
                for i in range(len(images)):
                    if predicted[i] != labels[i]:
                        misclassified_images.append({
                            'filename': img_path[i],
                            'label': labels[i].item(),
                            'score': scores[i].item(),
                            'predicted': predicted[i].item(),
                            'image': images[i].cpu()
                        })

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        cm = confusion_matrix(all_labels, all_predictions)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    avg_val_loss, val_losses, val_accuracy, train_losses, cm, misclassified_images

# Function to plot histogram and scatter plot using Bokeh
def plot_histogram_and_scatter(dataset, dc=None):
    output_notebook()

    # Prepare data for the histogram
    label_counts = Counter(dataset.labels)
    hist_data = {
        'labels': list(label_counts.keys()),
        'counts': list(label_counts.values())
    }
    hist_source = ColumnDataSource(data=hist_data)

    # Combine the data into a list of tuples (index, score, label)
    combined_data = list(zip(list(range(len(dataset.scores))), dataset.scores, dataset.labels))

    # Sort the combined data based on scores
    sorted_data = sorted(combined_data, key=lambda x: x[1])

    # Unpack the sorted data back into separate lists
    sorted_indices, sorted_scores, sorted_labels = zip(*sorted_data)

    # Prepare data for the scatter plot
    scatter_data = {
        'index': list(range(len(sorted_scores))),
        'scores': sorted_scores,
        'labels': sorted_labels
    }
    # print(scatter_data)
    scatter_source = ColumnDataSource(data=scatter_data)

    # Create the histogram
    hist_plot = figure(title="Count of Each Label", x_axis_label='Labels', y_axis_label='Count', tools="pan,wheel_zoom,box_zoom,reset,hover", tooltips=[("Label", "@labels"), ("Count", "@counts")])
    hist_plot.vbar(x='labels', top='counts', width=0.9, source=hist_source)
    hist_plot.width = 800
    hist_plot.height = 600

    # Show the histogram
    show(hist_plot, notebook_handle=True)

    # Create the scatter plot
    scatter_plot = figure(title="Scatter Plot of Scores", x_axis_label='Image Index', y_axis_label='Scores', tools="pan,wheel_zoom,box_zoom,reset,hover", tooltips=[("Index", "@index"), ("Score", "@scores"), ("Label", "@labels")])
    scatter_plot.circle('index', 'scores', size=10, source=scatter_source, alpha=0.5)
    scatter_plot.width = 800
    scatter_plot.height = 600

    # Show the scatter plot
    show(scatter_plot, notebook_handle=True)

    if dc is not None:
        dc.fetch_locals()

# Define the CNN model
class ChocolateCNN(nn.Module):
    def __init__(self):
        super(ChocolateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 25 * 6, 512) 
        # self.fc1 = nn.Linear(64 * 6 * 26, 512) 

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # x = x.view(-1, 64 * 6 * 26)  # Flatten the tensor
        x = x.view(-1, 64 * 25 * 6)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f'{val}', ha='center', va='center', color='red')

    plt.show()

# Function to plot training and validation losses
def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# Function to show misclassified images
def show_misclassified_images(misclassified_images, num_images=20):
    rows = (num_images + 3) // 4  # Calculate the number of rows needed
    plt.figure(figsize=(20, rows * 5))  # Adjust figure size accordingly

    for i, info in enumerate(misclassified_images[:num_images]):
        plt.subplot(rows, 4, i + 1)  # Arrange in a grid of 4 columns
        image = info['image'].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        # image = image.permute(1, 0, 2)
        # plt.imshow(image)
        plt.imshow(image)
        plt.title(f"File: {os.path.basename(info['filename'])}\n"
                  f"True: {info['label']}, Pred: {info['predicted']}\n"
                  f"Score: {info['score']}", fontsize=10)  # Smaller font size for filenames
        plt.axis('off')
    # plt.tight_layout()
    plt.show()

def show_misclassified_images1(misclassified_images, num_images=20):
    rows = (num_images + 3) // 4  # Calculate the number of rows needed
    fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 5))  # Adjust figure size accordingly

    for i, info in enumerate(misclassified_images[:num_images]):
        ax = axes[i // 4, i % 4]
        image = info['image'].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        ax.imshow(image, aspect='auto')  # Ensure aspect ratio is preserved
        ax.set_aspect('equal')
        ax.set_title(f"File: {os.path.basename(info['filename'])}\n"
                     f"True: {info['label']}, Pred: {info['predicted']}\n"
                     f"Score: {info['score']}", fontsize=10)  # Smaller font size for filenames
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Function to load the model
def load_model(model_path,device):
    model = ChocolateCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model1(model, dataloader, criterion, device):
    all_labels = []
    all_predictions = []
    misclassified_images = []
    running_val_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels, scores, img_path  in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for i in range(len(images)):
                if predicted[i] != labels[i]:
                    misclassified_images.append({
                        'filename': img_path[i],
                        'label': labels[i].item(),
                        'score': scores[i].item(),
                        'predicted': predicted[i].item(),
                        'image': images[i].cpu()
                    })

    avg_val_loss = running_val_loss / len(dataloader)
    val_accuracy = 100 * correct / total

    cm = confusion_matrix(all_labels, all_predictions)

    return cm, avg_val_loss, val_accuracy, misclassified_images


def save_model_with_metadata(
    model, optimizer, num_epochs, loss_function, avg_val_loss, val_accuracy,
    model_class_name, score_classification, val_losses, train_losses, 
    additional_text, val_indices, misclassified_images, cm, save_path
):
    # Create a dictionary to store metadata
    metadata = {
        'num_epochs': num_epochs,
        'loss_function': loss_function,
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': avg_val_loss,
        'valid_accuracy': val_accuracy,
        'model_class': model_class_name,
        'score_classification': score_classification,
        'val_losses': val_losses,
        'train_losses': train_losses,
        'additional_text': additional_text,
        'val_indices': val_indices,
        'misclassified_images': misclassified_images,
        'confusion_matrix': cm
    }

    # Save model state and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, save_path)


def load_model_with_metadata(load_path, device):
    # Load the model state and metadata
    checkpoint = torch.load(load_path)

    # Retrieve metadata
    metadata = checkpoint['metadata']

    # Dynamically get the model class
    model_class_name = metadata['model_class']
    # model_class = getattr(methods, model_class_name)
    model_class = globals()[model_class_name]
    model = model_class().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Retrieve other metadata
    num_epochs = metadata['num_epochs']
    loss_function = metadata['loss_function']
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(metadata['optimizer_state_dict'])
    additional_text = metadata['additional_text']
    valid_loss = metadata['valid_loss']
    valid_accuracy = metadata['valid_accuracy']
    score_classification = metadata['score_classification']
    val_indices = metadata['val_indices']
    misclassified_images = metadata['misclassified_images']
    cm = metadata['confusion_matrix']
    val_losses= metadata['val_losses']
    train_losses= metadata['train_losses']

    print(f'Number of Epochs: {num_epochs}')
    print(f'Loss Function: {loss_function}')
    print(f'Validation Loss: {valid_loss}')
    print(f'Validation Accuracy: {valid_accuracy}')
    print(f'Additional Text: {additional_text}')
    print(f'Model Class: {model_class_name}')
    print(f'Score Classification: {score_classification}')

    return (model, optimizer, num_epochs, loss_function, additional_text, valid_loss, 
            valid_accuracy, score_classification, misclassified_images, cm, val_indices, train_losses, val_losses)

def show_common_misclassified_images(common_misclassified_images, misclassified_details, model_paths, num_images=20):
    rows = (num_images + 3) // 4  # Calculate the number of rows needed
    plt.figure(figsize=(20, rows * 5))  # Adjust figure size accordingly

    for i, filename in enumerate(list(common_misclassified_images)[:num_images]):
        plt.subplot(rows, 4, i + 1)  # Arrange in a grid of 4 columns
        image = misclassified_details[model_paths[0]][filename]['image'].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        # image = image.permute(1, 0, 2)
        plt.imshow(image)
        title = f"File: {os.path.basename(filename)}"
        for model_path in model_paths:
            for detail in misclassified_details[model_path][filename]['details']:
                title += f"\n{os.path.basename(model_path)}, True: {detail['true']}, Pred: {detail['pred']}, Score: {detail['score']}"
        plt.title(title, fontsize=10)  
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_losses(num_epochs, train_losses, val_losses, title='Training and Validation Loss Over Epochs'):
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

