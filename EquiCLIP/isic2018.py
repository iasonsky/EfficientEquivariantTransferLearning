from PIL import Image
import csv
import os
from torch.utils.data import Dataset

# Dataset class for ISIC 2018 Task 3
class ISICDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = {}
        self.load_labels(label_file)

    def load_labels(self, label_file):
        with open(label_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip the first row, which is the header

            # Find the class columns in the header
            class_columns = header[1:]

            for row in reader:
                img_name = row[0]
                one_hot_labels = [float(x) for x in row[1:]]
                try:
                    # Find the index of the column containing 1.0 to map to a class index
                    class_idx = one_hot_labels.index(1.0)
                except ValueError:
                    class_idx = -1  # Default to -1 for unknown class
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