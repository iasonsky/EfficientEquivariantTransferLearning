import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import clip
import argparse
import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv

from load_model import load_model
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip

print("Torch version:", torch.__version__)
# Load environment variables
load_dotenv()

def main(args):
    # Set project and entity name
    project = os.getenv("WANDB_PROJECT", "dl-2024")
    entity = os.getenv("WANDB_ENTITY", "dl2-2024")

    # Initialize wandb
    wandb.init(project=project, entity=entity, config=vars(args))
    wandb.run.name = f"zs_{args.method}_{args.dataset_name}_{args.model_name}_{args.group_name}_{args.data_transformations}"
    # load model and preprocess
    model, preprocess = load_model(args)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)

    # get dataloader
    dataloader = get_dataloader(args, preprocess)

    # create text weights for different classes
    zeroshot_weights = zeroshot_classifier(args, model, classnames, templates, save_weights='True').to(args.device)

    # zeroshot prediction
    import time
    st_time = time.time()
    zeroshot_top1_acc, zeroshot_top5_acc, zeroshot_precision, zeroshot_recall, zeroshot_f1_score = eval_clip(args, model, zeroshot_weights, dataloader, data_transformations=args.data_transformations,
              group_name=args.group_name)
    wandb.log({"zeroshot_top1_acc": zeroshot_top1_acc, "zeroshot_top5_acc": zeroshot_top5_acc, "zeroshot_precision": zeroshot_precision,
                "zeroshot_recall": zeroshot_recall, "zeroshot_f1_score": zeroshot_f1_score})
    end_time = time.time()
    print(f"time taken: {end_time - st_time}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform zeroshot operation for various methods")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--logit_factor", default=1., type=float)
    parser.add_argument("--data_transformations", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--group_name", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--method", default="vanilla", type=str, help=["vanilla", "equitune", "equizero"])
    parser.add_argument("--model_name", default="RN50", type=str, help=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                        'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=["ImagenetV2", "CIFAR100", "ISIC2018", "MNIST"])
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--softmax", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')
    parser.add_argument("--undersample", action='store_true')
    parser.add_argument("--oversample", action='store_true')
    parser.add_argument("--validate_equivariance", action='store_true')
    parser.add_argument("--save_scores", action='store_true')
    args = parser.parse_args()

    args.verbose = True

    pl.seed_everything(args.seed)
    main(args)
