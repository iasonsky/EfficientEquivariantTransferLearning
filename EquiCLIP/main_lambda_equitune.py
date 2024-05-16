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

from tqdm.autonotebook import tqdm, trange
from weight_models import WeightNet, AttentionAggregation
from load_model import load_model
from weighted_equitune_utils import weighted_equitune_clip
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader, \
    get_ft_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip
from logging_setup import setup_logging

print("Torch version:", torch.__version__)


def main(args):
    # load model and preprocess
    model: CLIP
    model, preprocess = load_model(args)
    if args.use_underscore:
        model_, preprocess_ = load_model(args)
    else:
        model_, preprocess_ = None, None

    if args.method == "attention":
        feature_combination_module = AttentionAggregation(args)
    else:
        feature_combination_module = WeightNet(args)
    feature_combination_module.to(args.device)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)

    # get dataloader
    train_loader, eval_loader = get_ft_dataloader(args, preprocess)

    # optimizer and loss criterion
    criterion = nn.CrossEntropyLoss()

    if args.method == "attention":
        if args.prelr > 0.01:
            print("Attention model is being trained with a high learning rate. This is not recommended.")
        optimizer1 = optim.Adam(feature_combination_module.parameters(), lr=args.prelr)
    else:
        # only weight_net is trained not the model itself
        optimizer1 = optim.SGD(feature_combination_module.parameters(), lr=args.prelr, momentum=0.9)

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
        "model_": model_,
    }

    train_kwargs = val_kwargs.copy()
    train_kwargs["iter_print_freq"] = args.iter_print_freq
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

            # evaluating for only 50 steps using val=True
            top1 = eval_clip(
               args, model, zeroshot_weights, train_loader, val=True, **val_kwargs
            )

            if top1 > best_top1:
                best_top1 = top1
                best_model_weights = copy.deepcopy(feature_combination_module.state_dict())

            # finetune prediction
            model = weighted_equitune_clip(args, model, feature_combination_module, optimizer1, criterion, zeroshot_weights,
                                           train_loader, num_iterations=args.iter_per_prefinetune, **train_kwargs)

        torch.save(best_model_weights, MODEL_PATH)
        feature_combination_module.load_state_dict(torch.load(MODEL_PATH))

    # zeroshot eval on validation data
    print(f"Validation accuracy!")
    logging.info(f"Validation accuracy!")
    # val=True only for choosing the best lambda weights using the trainloader
    eval_clip(args, model, zeroshot_weights, eval_loader, val=True, **val_kwargs)

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
        eval_clip(args, model, zeroshot_weights, eval_loader, val=False, **val_kwargs)

    eval_clip(args, model, zeroshot_weights, eval_loader, val=False, **val_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weighted equituning')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--num_prefinetunes", default=20, type=int,
                        help="num of iterations for learning the lambda weights")
    parser.add_argument("--num_finetunes", default=8, type=int, help="number of finetune steps")
    parser.add_argument("--iter_per_prefinetune", default=100, type=int)
    parser.add_argument("--iter_per_finetune", default=500, type=int)
    parser.add_argument("--iter_print_freq", default=50, type=int)
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
    parser.add_argument("--use_underscore", action='store_true', 
        help=("If specified then use a separate frozen copy of the pretrained model as an input to the weight network even during finetuning of this model."
        "This matches the original implementation, but requires more VRAM."))
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')
    parser.add_argument("--undersample", action='store_true')
    parser.add_argument("--oversample", action='store_true')
    args = parser.parse_args()

    args.verbose = True

    pl.seed_everything(args.seed)
    setup_logging(args)
    main(args)

# python main_weighted_equitune.py  --dataset_name CIFAR100  --logit_factor 1.0  --iter_per_finetune 500 --method equitune --group_name rot90 --data_transformations rot90  --model_name 'ViT-B/16' --lr 0.000005 --num_finetunes 8 --num_prefinetunes 20 --k -10 --prelr 0.33
