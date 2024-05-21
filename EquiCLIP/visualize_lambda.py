import os

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import clip
import copy
import argparse
import os
from pathlib import Path
import pytorch_lightning as pl
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision

# ugly way of making import work when using as a Python module
# equiclip_path = os.path.join(os.path.dirname(__file__), "EquiCLIP")
equiclip_path = os.path.dirname(__file__)
assert "EquiCLIP" in equiclip_path
if equiclip_path not in sys.path:
    sys.path.append(equiclip_path)
# print(sys.path)

from tqdm import tqdm
from pkg_resources import packaging
from weight_models import WeightNet
from load_model import load_model
from weighted_equitune_utils import weighted_equitune_clip
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader, get_ft_dataloader, get_ft_visualize_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

print("Torch version:", torch.__version__)

# observed lambda_values
# RN50
def plot_lambda_weights():
    x = [0, 1, 2, 3]
    lambda_rn50 = torch.tensor([101.3, 279.3, 164., 347.5])
    lambda_rn50 = lambda_rn50 / sum(lambda_rn50)
    lambda_rn101 = torch.tensor([18.92, 21.14, 24.08, 27.2])
    lambda_rn101 = lambda_rn101 / sum(lambda_rn101)
    lambda_vit32 = torch.tensor([42.44, 30.75, 44.53, 57.47])
    lambda_vit32 = lambda_vit32 / sum(lambda_vit32)
    lambda_vit16 = torch.tensor([244.9, 437.8, 374.3, 366.8])
    lambda_vit16 = lambda_vit16 / sum(lambda_vit16)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Weighted equituning')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--img_num", default=0, type=int)
    parser.add_argument("--num_prefinetunes", default=10, type=int, help="num of iterations for learning the lambda weights")
    # parser.add_argument("--num_finetunes", default=8, type=int)
    # parser.add_argument("--iter_per_prefinetune", default=100, type=int)
    # parser.add_argument("--iter_per_finetune", default=500, type=int)
    parser.add_argument("--data_transformations", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--group_name", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--method", default="equitune", type=str, help=["vanilla", "equitune", "equizero"])
    parser.add_argument("--model_name", default="RN50", type=str, help=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                        'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=["ImagenetV2", "CIFAR100"])
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--softmax", action='store_true')
    parser.add_argument("--use_underscore", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')
    parser.add_argument("--visualize_features", action='store_true',
        help="Visualize intermediate features on top of the lambda weights")
    parser.add_argument("--model_file", default="", type=str, help="File name of the model. If set then other parameters are discarded.")
    parser.add_argument("--output_filename_suffix", default="", type=str, help="File name suffix of the output dataframe. Specify it to avoid name clashes when generating plots with multiple input models where the parameters are not unique")
    parser.add_argument("--model_display_name", default="", type=str, help="")
    args = parser.parse_args(argv)

    args.verbose = True

    pl.seed_everything(args.seed)

    if not args.model_display_name:
        args.model_display_name = args.model_name

    return args

def main(args):
    args = parse_args(args)
    print(args)
    # load model and preprocess
    model, preprocess = load_model(args)

    # load weight network
    weight_net = WeightNet(args).to(args.device)

    # get dataloader
    train_loader, eval_loader = get_ft_dataloader(args, preprocess)
    viz_train_loader, viz_eval_loader = get_ft_visualize_dataloader(args, preprocess)

    if not args.model_file:
        MODEL_DIR = "saved_weight_net_models"

        if args.model_name in ['ViT-B/32', 'ViT-B/16']:
            args.save_model_name = ''.join(args.model_name.split('/'))
        else:
            args.save_model_name = args.model_name

        MODEL_NAME = f"{args.dataset_name}_{args.save_model_name}_aug_{args.data_transformations}_eq_{args.group_name}" \
                    f"_steps_{args.num_prefinetunes}.pt"
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    else:
        MODEL_PATH = args.model_file
        MODEL_NAME = Path(MODEL_PATH).name
        MODEL_DIR = Path(MODEL_PATH).parent
        args.save_model_name = args.model_name

    if args.method == "attention":
        feature_combination_module = AttentionAggregation(args)
    else:
        feature_combination_module = WeightNet(args)
    feature_combination_module.to(args.device)

    if os.path.isfile(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        weight_net.load_state_dict(torch.load(MODEL_PATH))
    else:
        raise Exception(f"Please train a model first (using main_lambda_equitune.py)")

    all_weights = []
    # compute lambda for different transformed images
    writer = SummaryWriter()
    for i, data in enumerate(tqdm(eval_loader)):
        x, y = data
        x = x.to(args.device)
        x_group = []
        weights_for_all_trafos = []
        for j in range(4):
            x = torch.rot90(x, k=1, dims=(-1, -2))
            x_group.append(x)
            if args.visualize_features:
                # Image 'features' here are really the image embeddings produced by CLIP, which are invariant
                # `internal_features` are the output of the last convolution layer of the backbone, 
                # where we expect to see actual equivariance
                image_features, internal_features = model.encode_image(x, return_internal_features=True)  # dim [group_size * batch_size, feat_size=512]
                # print(internal_features.shape, internal_features.dtype)
                for k in range(len(internal_features)):
                    grid = torchvision.utils.make_grid(internal_features[k].unsqueeze(1), normalize=False, nrow=64)
                    writer.add_image(f'internal features{k}', grid, j)

                    fn = f"feature_visualizations/{args.dataset_name}_{args.save_model_name}_aug_{args.data_transformations}_eq_{args.group_name}{args.output_filename_suffix}_batch_{i}_image_{k}_group_{j}.png"
                    torchvision.utils.save_image(grid, fn)
            else:
                image_features = model.encode_image(x)  # dim [group_size * batch_size, feat_size=512]
            weights = weight_net(image_features.float()).half()
            assert weights.shape[-1] == 1
            for k in range(len(weights)):
               weight = weights[k]
               writer.add_scalar(f'lambda{k}', weight, j)
            weights_for_all_trafos.append(weights[:, 0].detach().cpu().numpy())
        weights_for_all_trafos = np.stack(weights_for_all_trafos)
        all_weights.append(weights_for_all_trafos)
        # raise Exception("e")

    # Save actual images to Tensorboard - this use a different data loader, skipping some preprocessing steps,
    # that's why it is a separate loop
    for i, data in enumerate(viz_eval_loader):
        x, y = data
        x = x.to(args.device)
        for j in range(4):
            x = torch.rot90(x, k=1, dims=(-1, -2))

            for k in range(len(x)):
                grid = torchvision.utils.make_grid(x[k])
                writer.add_image(f'images{k}', grid, j)

        break
    writer.close()

    all_weights = np.concatenate(all_weights, axis=-1).astype(np.float32)
    df = pd.DataFrame(all_weights.T, columns=["90", "180", "270", "0"]).loc[:, ["0", "90", "180", "270"]]
    df["model_name"] = args.model_name
    df["model_display_name"] = args.model_display_name
    df["dataset_name"] = args.dataset_name
    df["group_name"] = args.group_name
    df["data_transformations"] = args.data_transformations
    df["full_finetune"] = args.full_finetune
    df["method"] = args.method

    output_dir = "results/lambda_weights"
    os.makedirs(output_dir, exist_ok=True)
    # np.save(f"{output_dir}/lambda_weights_{MODEL_NAME}.npy", all_weights)
    df.to_csv(f"{output_dir}/lambda_weights_{MODEL_NAME}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])

# python main_weighted_equitune.py  --dataset_name CIFAR100  --logit_factor 1.0  --iter_per_finetune 500 --method equitune --group_name rot90 --data_transformations rot90  --model_name 'ViT-B/16' --lr 0.000005 --num_finetunes 10 --num_prefinetunes 20 --k -10 --prelr 0.33