import os

from clip.model import CLIP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import clip
import copy
import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from dotenv import load_dotenv

from tqdm.autonotebook import trange
from weight_models import WeightNet, AttentionAggregation
from load_model import load_model
from weighted_equitune_utils import weighted_equitune_clip
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader, \
    get_ft_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip
from logging_setup import setup_logging

print("Torch version:", torch.__version__)
# Load environment variables
load_dotenv()

def main(args):
    # Set project and entity name
    project = os.getenv("WANDB_PROJECT", "dl-2024")
    entity = os.getenv("WANDB_ENTITY", "dl2-2024")

    # Initialize wandb
    wandb.init(project=project, entity=entity, config=vars(args), tags=["equivariant features"])
    wandb.run.name = f"lambda_{args.method}_{args.dataset_name}_{args.model_name}_lr{args.lr}_{args.group_name}_{args.data_transformations}"
    # load model and preprocess
    model: CLIP
    model, preprocess = load_model(args)

    if args.method == "attention":
        feature_combination_module = AttentionAggregation(args.model_name)
    else:
        feature_combination_module = WeightNet(args)
    feature_combination_module.to(args.device)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)

    # get dataloader
    train_loader, eval_loader = get_ft_dataloader(args, preprocess)

    # optimizer and loss criterion
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.method == "attention":
        if args.prelr > 0.01:
            print("Attention model is being trained with a high learning rate. This is not recommended.")
        optimizer1 = optim.SGD(feature_combination_module.parameters(), lr=args.prelr, momentum=0.9)
        # optimizer1 = optim.Adam(feature_combination_module.parameters(), lr=args.prelr, eps=1e-5)
        # warmup scheduler
        # lr_scheduler = optim.lr_scheduler.LinearLR(optimizer1, start_factor=0.01, total_iters=100)
    else:
        # only weight_net is trained not the model itself
        optimizer1 = optim.SGD(feature_combination_module.parameters(), lr=args.prelr, momentum=0.9)

    temp = f"Optimizing {sum(p.numel() for p in feature_combination_module.parameters())} parameters."
    print(temp)
    logging.info(temp)

    # create text weights for different classes
    zeroshot_weights = zeroshot_classifier(args, model, classnames, templates, save_weights='True').to(args.device)

    best_top1 = 0.0
    best_model_weights = copy.deepcopy(feature_combination_module.state_dict())
    MODEL_DIR = "saved_weight_net_models"

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if args.model_name in ['ViT-B/32', 'ViT-B/16']:
        args.save_model_name = ''.join(args.model_name.split('/'))
    else:
        args.save_model_name = args.model_name

    MODEL_NAME = f"{args.dataset_name}_{args.save_model_name}_aug_{args.data_transformations}_eq_{args.group_name}" \
                 f"_steps_{args.num_prefinetunes}.pt"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    val_kwargs = {
        "data_transformations": args.data_transformations,
        "group_name": args.group_name,
        "device": args.device,
        "feature_combination_module": feature_combination_module,
    }

    train_kwargs = val_kwargs.copy()
    del train_kwargs["feature_combination_module"]

    if os.path.isfile(MODEL_PATH) and args.load:
        feature_combination_module.load_state_dict(torch.load(MODEL_PATH))
    else:
        for i in trange(args.num_prefinetunes, desc="Pre-fine tunes"):
            if args.method == "attention":
                print(f"Learning attention weights: {i}/{args.num_prefinetunes}")
            else:
                print(f"Learning lambda weights: {i}/{args.num_prefinetunes}")
            # zeroshot prediction
            # add weight_net save code for the best model
            val = False if args.full_val_pf else True # evaluating for only 50 steps using val=True if full_val_pf is False
            prefinetune_top1_acc, prefinetune_top5_acc, prefinetune_precision, prefinetune_recall, prefinetune_f1_score = eval_clip(
                args, model, zeroshot_weights, train_loader, val=val, **val_kwargs
            )
            wandb.log({"prefinetune_top1_acc": prefinetune_top1_acc, "prefinetune_top5_acc": prefinetune_top5_acc, "prefinetune_precision": prefinetune_precision, 
                    "prefinetune_recall": prefinetune_recall, "prefinetune_f1_score": prefinetune_f1_score})
            if prefinetune_top1_acc > best_top1:
                best_top1 = prefinetune_top1_acc
                best_model_weights = copy.deepcopy(feature_combination_module.state_dict())

            # finetune prediction
            model = weighted_equitune_clip(
                args, model, feature_combination_module, optimizer1, criterion, zeroshot_weights,
                train_loader, num_iterations=args.iter_per_prefinetune, lr_scheduler=lr_scheduler,
                **train_kwargs
            )

        torch.save(best_model_weights, MODEL_PATH)
        feature_combination_module.load_state_dict(torch.load(MODEL_PATH))

    # zeroshot eval on validation data
    print(f"Validation accuracy!")
    logging.info(f"Validation accuracy!")
    # val=True only for choosing the best lambda weights using the trainloader
    val_top1_acc, val_top5_acc, val_precision, val_recall, val_f1_score = eval_clip(
        args, model, zeroshot_weights, eval_loader, val=False, **val_kwargs
    )
    wandb.log({"val_top1_acc": val_top1_acc, "val_top5_acc": val_top5_acc, "val_precision": val_precision, "val_recall": val_recall, "val_f1_score": val_f1_score})

    # Save the weighting model as an artifact
    artifact = wandb.Artifact('Weighting_model', type='model')
    artifact.add_file(MODEL_PATH)
    wandb.log_artifact(artifact)

    if args.full_finetune:
        optimizer2 = optim.SGD(list(model.parameters()) + list(feature_combination_module.parameters()), lr=args.lr, momentum=0.9)
    else:
        optimizer2 = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for i in trange(args.num_finetunes, desc="Fine tunes"):
        print(f"Model finetune step number: {i}/{args.num_finetunes}")
        logging.info(f"Model finetune step number: {i}/{args.num_finetunes}")

        model = weighted_equitune_clip(args, model, feature_combination_module,
                                       optimizer2, criterion, zeroshot_weights, train_loader,
                                       num_iterations=args.iter_per_finetune,
                                       **train_kwargs)
        finetune_top1_acc, finetune_top5_acc, finetune_precision, finetune_recall, finetune_f1_score = eval_clip(
            args, model, zeroshot_weights, eval_loader, val=False, **val_kwargs
        )
        wandb.log({"finetune_top1_acc": finetune_top1_acc, "finetune_top5_acc": finetune_top5_acc, "finetune_precision": finetune_precision,
                    "finetune_recall": finetune_recall, "finetune_f1_score": finetune_f1_score})
    final_top1_acc, final_top5_acc, final_precision, final_recall, final_f1_score = eval_clip(
        args, model, zeroshot_weights, eval_loader, val=False, **val_kwargs
    )
    wandb.log({"final_top1_acc": final_top1_acc, "final_top5_acc": final_top5_acc, "final_precision": final_precision, "final_recall": final_recall, "final_f1_score": final_f1_score})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weighted equituning')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--num_prefinetunes", default=20, type=int,
                        help="num of iterations for learning the lambda weights")
    parser.add_argument("--num_finetunes", default=8, type=int, help="number of finetune steps")
    parser.add_argument("--iter_per_prefinetune", default=100, type=int)
    parser.add_argument("--iter_per_finetune", default=500, type=int)
    parser.add_argument("--logit_factor", default=1., type=float)
    parser.add_argument("--prelr", default=0.33, type=float)
    parser.add_argument("--lr", default=0.000005, type=float)
    parser.add_argument("--data_transformations", default="", type=str, help=str(["", "flip", "rot90"]))
    parser.add_argument("--group_name", default="", type=str, help=str(["", "flip", "rot90"]))
    parser.add_argument("--method", default="equitune", type=str,
                        help=str(["vanilla", "equitune", "equizero", "attention"]))
    parser.add_argument("--model_name", default="RN50", type=str, help=str(['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                            'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                            'ViT-L/14', 'ViT-L/14@336px']))
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=str(["ImagenetV2", "CIFAR100", "ISIC2018", "MNIST"]))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--softmax", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')
    parser.add_argument("--undersample", action='store_true')
    parser.add_argument("--oversample", action='store_true')
    parser.add_argument("--kaggle", action='store_true')
    parser.add_argument("--full_val_pf", action='store_true')
    parser.add_argument("--validate_equivariance", action='store_true')
    args = parser.parse_args()

    args.verbose = True

    pl.seed_everything(args.seed)
    setup_logging(args)
    main(args)

# python main_weighted_equitune.py  --dataset_name CIFAR100  --logit_factor 1.0  --iter_per_finetune 500 --method equitune --group_name rot90 --data_transformations rot90  --model_name 'ViT-B/16' --lr 0.000005 --num_finetunes 8 --num_prefinetunes 20 --k -10 --prelr 0.33
