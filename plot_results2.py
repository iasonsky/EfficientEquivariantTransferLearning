import matplotlib.pyplot as plt
import numpy as np
import os

def load_accuracies_from_files(dataset, model_names, method, transformation):
    accuracies = []
    base_path = f"results/{dataset}"
    for model in model_names:
        file_path = os.path.join(base_path, model, method, transformation, 'top1_accuracy.txt')
        try:
            with open(file_path, 'r') as file:
                accuracy = float(file.read().strip())
                accuracies.append(accuracy)
        except FileNotFoundError:
            # If the specific result file is not found, append a None or a placeholder value (e.g., 0)
            accuracies.append(None)
    return np.array(accuracies)

def plot_1_imagenet_rot90(models, save_fig=False):
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), models, fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    # Load the accuracies from files instead of hardcoding
    y_original = load_accuracies_from_files('ImagenetV2', models, 'vanilla', 'rot90')
    y_equitune = load_accuracies_from_files('ImagenetV2', models, 'equitune', 'rot90')
    y_equizero = load_accuracies_from_files('ImagenetV2', models, 'equizero', 'rot90')

    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")

    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    plt.ylim([35, 65])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-Imagenet V2", fontsize=15)
    plt.tight_layout()

    if save_fig:
        plt.savefig("plot_1_imagenet_rot90.png", dpi=150)

    plt.show()
    plt.close()

def plot_1_imagenet_flip(models, save_fig=False):
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), models, fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = load_accuracies_from_files('ImagenetV2', models, 'vanilla', 'flip')
    y_equitune = load_accuracies_from_files('ImagenetV2', models, 'equitune', 'flip')
    y_equizero = load_accuracies_from_files('ImagenetV2', models, 'equizero', 'flip')

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    plt.ylim([40, 65])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for flip-Imagenet V2", fontsize=15)
    plt.tight_layout()

    if save_fig:
        plt.savefig("plot_1_imagenet_flip.png", dpi=150)

    plt.show()
    plt.close()

def plot_1_cifar100_rot90(models, save_fig=False):
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), models, fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = load_accuracies_from_files('CIFAR100', models, 'vanilla', 'rot90')
    y_equitune = load_accuracies_from_files('CIFAR100', models, 'equitune', 'rot90')
    y_equizero = load_accuracies_from_files('CIFAR100', models, 'equizero', 'rot90')

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    plt.ylim([22, 70])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-CIFAR100", fontsize=15)
    plt.tight_layout()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        plt.savefig("plot_1_cifar100_rot90.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    plt.show()
    plt.close()

def plot_1_cifar100_flip(models, save_fig=False):
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), models, fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = load_accuracies_from_files('CIFAR100', models, 'vanilla', 'flip')
    y_equitune = load_accuracies_from_files('CIFAR100', models, 'equitune', 'flip')
    y_equizero = load_accuracies_from_files('CIFAR100', models, 'equizero', 'flip')

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([27.5, 68.5])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-CIFAR100", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        plt.savefig("plot_1_cifar100_flip.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    plt.close()

# Assuming the following are the models we are interested in.
models = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']
plot_1_imagenet_rot90(models, save_fig=True)
plot_1_imagenet_flip(models, save_fig=True) 
plot_1_cifar100_rot90(models, save_fig=True)
plot_1_cifar100_flip(models, save_fig=True)
