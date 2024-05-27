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
from weight_models import WeightNet, AttentionAggregation
from load_model import load_model
from weighted_equitune_utils import weighted_equitune_clip
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader, get_ft_dataloader, get_ft_visualize_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip
from torch.utils.tensorboard import SummaryWriter
from weighted_equitune_utils import compute_logits
from exp_utils import group_transform_images, random_transformed_images, inverse_transform_images, verify_invariance, \
    verify_weight_equivariance


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
    parser.add_argument("--method", default="equitune", type=str,
                        help=str(["vanilla", "equitune", "equizero", "attention"]))
    parser.add_argument("--model_name", default="RN50", type=str, help=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                        'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=["ImagenetV2", "CIFAR100"])
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--softmax", action='store_true')
    parser.add_argument("--use_underscore", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--full_finetune", action='store_true')
    # parser.add_argument("--visualize_features", action='store_true',
    #     help="Visualize intermediate features on top of the lambda weights")
    parser.add_argument("--model_file", default="", type=str, help="File name of the model. If set then other parameters are discarded.")
    parser.add_argument("--output_filename_suffix", default="", type=str, help="File name suffix of the output dataframe. Specify it to avoid name clashes when generating plots with multiple input models where the parameters are not unique")
    parser.add_argument("--model_display_name", default="", type=str, help="")
    parser.add_argument("--undersample", action='store_true')
    parser.add_argument("--oversample", action='store_true')
    parser.add_argument("--kaggle", action='store_true')
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

    # get dataloader
    train_loader, eval_loader = get_ft_dataloader(args, preprocess, batch_size=8)
    # viz_train_loader, viz_eval_loader = get_ft_visualize_dataloader(args, preprocess, batch_size=1)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)
    # create text weights for different classes
    zeroshot_weights = zeroshot_classifier(args, model, classnames, templates, save_weights='True').to(args.device)

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
        feature_combination_module = AttentionAggregation(args.model_name)
    else:
        feature_combination_module = WeightNet(args)
    feature_combination_module.to(args.device)

    if os.path.isfile(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        feature_combination_module.load_state_dict(torch.load(MODEL_PATH))
    else:
        raise Exception(f"Please train a model first (using main_lambda_equitune.py)")

    all_weights = []
    # compute lambda for different transformed images
    writer = SummaryWriter()
    for i, data in enumerate(tqdm(eval_loader)):
        images, target = data
        # weights_for_all_trafos = []
    
        images = images.to(args.device)  # dim [batch_size, c_in, H, H]
        # images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

        group_images = group_transform_images(images,
                                              group_name=args.group_name)  # dim [group_size, batch_size, c_in, H, H]
        group_images_shape = group_images.shape

        # dim [group_size * batch_size, c_in, H, H]
        group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                            group_images_shape[3], group_images_shape[3])
        
        logits, weights = compute_logits(args=args,
                                         model=model,
                                         feature_combination_module=feature_combination_module,
                                         group_images=group_images,
                                         zeroshot_weights=zeroshot_weights,
                                         group_name=args.group_name,
                                         validate_equivariance=False,  # here it is a separate arg because it is only called in validation
                                         return_weights=True,
                                         log_variance=False
                                         )
        if args.method == "attention":
            assert weights.shape == torch.Size([images.shape[0], 4, 4])
            # Attention weights are square, [B, G, G], as they contain one value for each combination of
            # source_feature_map, destination_feature_map.
            # However, after being multiplied with the features, the
            # mean of the result is taken, so I take the mean of them here for plotting purposes.
            # This basically averages how much a given feature map is "attended to" by all the other ones.
            # The direction is important here, taking the mean in the other direction would result in
            # [0.25, 0.25, 0.25, 0.25] always.
            weights = weights.mean(dim=1)
        else:
            # Weights are of shape [B, G, 1, 1, 1], the last 3 dimensions were only added internally to make
            # them compatbile with the (group-transformation-expanded) image features.
            assert weights.shape == torch.Size([images.shape[0], 4, 1, 1, 1])
            weights = weights[:, :, 0, 0, 0]
        #weights: [B, G]
        assert weights.shape == torch.Size([images.shape[0], 4])
        
        for k in range(len(weights)): # batch
            for j in range(len(weights[k])): # group
                weight = weights[k, j]
                writer.add_scalar(f'lambda{k}', weight, j)
        
        # if args.visualize_features:
        #     # FIXME not yet updated to the latest master
        #     for k in range(len(internal_features)):
        #         grid = torchvision.utils.make_grid(internal_features[k].unsqueeze(1), normalize=False, nrow=64)
        #         writer.add_image(f'internal features{k}', grid, j)
        #
        #         fn = f"feature_visualizations/{args.dataset_name}_{args.save_model_name}_aug_{args.data_transformations}_eq_{args.group_name}{args.output_filename_suffix}_batch_{i}_image_{k}_group_{j}.png"
        #         torchvision.utils.save_image(grid, fn)

        all_weights.append(weights[:, :].detach().cpu().numpy())

    # Save actual images to Tensorboard - this use a different data loader, skipping some preprocessing steps,
    # that's why it is a separate loop
    # for i, data in enumerate(viz_eval_loader):
    #     images, target = data
    
    #     images = images.to(args.device)  # dim [batch_size, c_in, H, H]
    #     group_images = group_transform_images(images,
    #                                           group_name=args.group_name)  # dim [group_size, batch_size, c_in, H, H]
    #     assert group_images.shape[1] == 1, "Only batches of 1 are supported"
    #     grid = torchvision.utils.make_grid(group_images[:, 0, :, :, :], normalize=False, nrow=1)
    #     torchvision.utils.save_image(grid, f"feature_visualizations/input_{i}.png")

    #     if i > 6:
    #         break

    writer.close()

    all_weights = np.concatenate(all_weights, axis=0).astype(np.float32)
    df = pd.DataFrame(all_weights, columns=["0", "90", "180", "270"])
    df["model_name"] = args.model_name
    df["model_display_name"] = args.model_display_name
    df["dataset_name"] = args.dataset_name
    df["group_name"] = args.group_name
    df["data_transformations"] = args.data_transformations
    df["full_finetune"] = args.full_finetune
    df["method"] = args.method

    output_dir = "results/lambda_weights"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/lambda_weights_{MODEL_NAME}_{args.output_filename_suffix}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])

# python main_weighted_equitune.py  --dataset_name CIFAR100  --logit_factor 1.0  --iter_per_finetune 500 --method equitune --group_name rot90 --data_transformations rot90  --model_name 'ViT-B/16' --lr 0.000005 --num_finetunes 10 --num_prefinetunes 20 --k -10 --prelr 0.33