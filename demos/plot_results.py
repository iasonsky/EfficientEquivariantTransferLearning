import os
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the folder structure and that the top-1 accuracy is stored in 'top1_accuracy.txt'
base_folder = "results"  # Base folder where results are stored

# Placeholder for dataset, model, and method names
datasets = ["ImagenetV2", "CIFAR100"]
models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']
methods = ["", "flip", "rot90"]  # Assuming "" denotes no transformation

# Data structure to hold the accuracies
accuracies = {dataset: {model: {} for model in models} for dataset in datasets}

# Iterate over each dataset, model, and method to read the accuracies
for dataset in datasets:
    for model in models:
        for method in methods:
            # Construct the file path for top1_accuracy.txt
            method_folder = "" if method == "" else method
            folder_path = os.path.join(base_folder, dataset, model, "vanilla", method_folder)
            file_path = os.path.join(folder_path, "top1_accuracy.txt")

            # Read the accuracy value if the file exists
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    accuracy = float(file.readline().strip())
                    accuracies[dataset][model][method] = accuracy
            else:
                # If the file doesn't exist, we can either skip or set a default value
                accuracies[dataset][model][method] = None  # Or some default value like 0

# Function to create bar plot
def create_bar_plot(data, title):
    labels = models
    no_transformation = [data[model][""] for model in models]
    flip = [data[model]["flip"] for model in models]
    rot90 = [data[model]["rot90"] for model in models]
    
    x = range(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, no_transformation, width, label='No transformation')
    rects2 = ax.bar([p + width for p in x], flip, width, label='Random flips')
    rects3 = ax.bar([p + width * 2 for p in x], rot90, width, label='Random 90\' rotations')

    ax.set_ylabel('Top-1 accuracy')
    ax.set_title(title)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylim([35, 65]) if dataset == "ImagenetV2" else plt.ylim([22, 70])
    plt.show()
    plt.savefig(f"top1_accuracy_{dataset}.png")

# Create bar plots for each dataset
for dataset in datasets:
    create_bar_plot(accuracies[dataset], f"Top-1 accuracy for {dataset}")

