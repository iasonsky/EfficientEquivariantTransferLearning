import os
import csv
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define a dataset class for ISIC 2018 Task 3
class ISICDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = {}
        self.classes = []
        self.load_labels(label_file)

    def load_labels(self, label_file):
        # Read the CSV file
        with open(label_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # First row is the header
            self.classes = header[1:]  # Ignore the first column ('image')

            # Iterate through the remaining rows to map images to their labels
            for row in reader:
                img_name = row[0]
                one_hot_labels = [float(x) for x in row[1:]]
                try:
                    class_idx = one_hot_labels.index(1.0)  # Find the index with value 1.0
                except ValueError:
                    class_idx = -1  # If no 1.0 is found, default to -1 (unknown class)
                self.labels[img_name] = class_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.labels[img_name]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_image_by_class(self, class_idx):
        for img_name, label in self.labels.items():
            if label == class_idx:
                img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
        return None, None

def collect_images_by_class(dataset):
    images = []
    labels = []
    for class_idx in range(len(dataset.classes)):
        image, label = dataset.get_image_by_class(class_idx)
        if image is not None:
            images.append(image)
            labels.append(label)
    return images, labels

# Define the dataset and transformation
train_img_dir = 'data/ISIC2018/ISIC2018_Task3_Training_Input'
train_label_file = 'data/ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

isic_dataset = ISICDataset(train_img_dir, train_label_file, transform=transform)

# Collect one image per class
images, labels = collect_images_by_class(isic_dataset)
class_names = isic_dataset.classes

# Plot the images
def show_images(images, labels, class_names):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 20))
    for img, lbl, ax in zip(images, labels, axes):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(class_names[lbl])
        ax.axis('off')
    plt.savefig('isic2018.png')
    plt.show()

show_images(images, labels, class_names)

import matplotlib.pyplot as plt
from collections import Counter

# Assuming your dataset class is already created
train_label_file = 'data/ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv'
isic_dataset = ISICDataset(img_dir='data/ISIC2018/ISIC2018_Task3_Training_Input', label_file=train_label_file)

# Count the class distribution
class_counts = Counter(isic_dataset.labels.values())

# Map the counts to class names for readability
class_distribution = {isic_dataset.classes[i]: class_counts[i] for i in range(len(isic_dataset.classes))}

# Plot the distribution
plt.bar(class_distribution.keys(), class_distribution.values())
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in ISIC 2018 Dataset')
plt.xticks(rotation=45)
plt.show()

# Print raw counts
print("Class distribution:", class_distribution)
