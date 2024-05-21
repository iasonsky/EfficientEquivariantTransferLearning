from typing import Optional, Union

import torch
import torch.nn.functional as F
from clip.model import CLIP
import wandb
from tqdm.autonotebook import tqdm, trange
from itertools import cycle
from weight_models import AttentionAggregation, WeightNet
from exp_utils import group_transform_images, random_transformed_images

group_sizes = {"rot90": 4., "flip": 2., "": 1.}

# def cycle(iterable):
#     # this does not reset the iterable,
#     # so it hangs the process with an infinite loop when iterable reaches the end (i think)
#     while True:
#         for x in iterable:
#             yield x


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def get_output(output, group_name="", reduction="mean"):
    if group_name == "":
        return output
    elif group_name == "rot90":
        group_size = 4
    elif group_name == "flip":
        group_size = 2
    else:
        raise NotImplementedError
    
    output_shape = output.shape
    output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
    
    if reduction == "mean":
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
    elif reduction == "max":
        output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
    else:
        raise ValueError("Unsupported reduction type. Use 'mean' or 'max'.")
    
    return output

def compute_logits(args,
                   feature_combination_module: Union[WeightNet, AttentionAggregation],
                   image_features, image_features_,
                   zeroshot_weights,
                   group_size):
    if getattr(args, 'use_underscore', False) is False:
        image_features_ = image_features
    if args.method == "attention":
        # to B, N, D form
        image_features = image_features.view(-1, group_size, image_features.shape[-1])
        # dim [batch_size, feat_size]
        # or [group_size * batch_size, feat_size] if we run their weird logit averaging setup
        combined_features = feature_combination_module(image_features.float()).half()  # dim [batch_size, feat_size]
        logits = combined_features @ zeroshot_weights
    elif args.method == "vanilla" or feature_combination_module is None:
        logits = args.logit_factor * image_features @ zeroshot_weights
        if args.softmax:
            logits = torch.nn.functional.softmax(logits, dim=-1)
    else:
        # weighted image features
        # use .half since the model is in fp16
        # normalize group weights proportional to size of group_size
        group_weights = feature_combination_module(image_features_.float()).half()  # dim [group_size * batch_size, feat_size]
        # but that should reduce the dim to [group_size * batch_size, 1], no? then that k in einsum makes sense

        # group_weights = group_weights.reshape(group_images_shape[0], -1, 1)
        # group_weights = F.softmax(group_weights, dim=0)
        # weight_sum = torch.sum(group_weights, dim=0, keepdim=True)
        # print(f"weight_sum: {weight_sum}")
        # print(f"group weights: {group_weights.permute(1, 0, 2)}")
        # group_size = group_sizes[args.group_name]
        # group_weights = group_size * (group_weights / weight_sum)
        # group_weights = group_weights.reshape(-1, 1)

        # image_features = image_features_ * torch.broadcast_to(group_weights, image_features_.shape)
        # i think this is the same as the einsum, but lets stick to the original code
        image_features = torch.einsum('ij, ik -> ij', image_features.clone(), group_weights)

        # zeroshot weights correspond to text features for all possible classes
        # logits = 100. * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]

        # IMPORTANT NOTE: higher logit factors automatically biases the model towards the one with higher scores, hence,
        # acts like (un)equituning naturally even without lambda
        logits = args.logit_factor * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]

        if args.softmax:
            logits = torch.nn.functional.softmax(logits, dim=-1)
    return logits

def weighted_equitune_clip(args, model: CLIP,
                           feature_combination_module: Union[WeightNet, AttentionAggregation],
                           optimizer, criterion,
                           zeroshot_weights, loader,
                           data_transformations="", group_name="",
                           num_iterations=100, iter_print_freq=10, device="cuda:0",
                           model_=None):
    """
    Trains either model (clip), or weightnet, or both, depending on the optimizer
    Args:
        args:
        model: clip with the model_name that you specified. Here, it only runs the vision encoder (like RN50)
        feature_combination_module: a NN module that takes all features from a group
        optimizer:
        criterion:
        zeroshot_weights: embeddings of text prompts
        loader:
        data_transformations:
        group_name:
        num_iterations: steps to train
        iter_print_freq:
        device:
        model_:

    Returns:

    """
    import time
    torch.autograd.set_detect_anomaly(True)
    since = time.time()
    top1, top5, n = 0., 0., 0.
    training_iterator = cycle(iter(loader))
    # for i, (images, target) in enumerate(tqdm(loader)):
    import time
    st_time = time.time()
    for i in trange(num_iterations, desc="Training CLIP and/or WeightNet"):
        # if (i+1)%iter_print_freq == 0:
        #     print(f"iteration number: {i+1}")
        #     curr_time = time.time()
        #     print(f"time elapsed per iter: {(curr_time - st_time) / (i + 1)}")
        (images, target) = next(training_iterator)
        images = images.to(device)  # dim [batch_size, c_in, H, H]
        images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

        # images = torch.rot90(images, k=1, dims=(-2, -1))
        group_images = group_transform_images(images,
                                              group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
        group_images_shape = group_images.shape

        # dim [group_size * batch_size, c_in, H, H]
        group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                            group_images_shape[3], group_images_shape[3])
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict
        image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
        if not model_ is None:
            image_features_ = model_.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]

        # print(f"image_features.shape: {image_features.shape}")
        image_features_norm = image_features.clone().norm(dim=-1, keepdim=True)
        image_features = image_features / image_features_norm

        if not model_ is None:
            image_features_norm_ = image_features_.clone().norm(dim=-1, keepdim=True)
            image_features_ = image_features_ / image_features_norm_
        else:
            image_features_ = None  # filled in by compute_logits when it is empty

        logits = compute_logits(args, feature_combination_module, image_features, image_features_,
                                zeroshot_weights, group_images_shape[0])

        # measure accuracy
        if args.method == "equitune":
            output = get_output(logits, group_name=group_name, reduction="mean")
        elif args.method == "equizero":
            equitune_output = get_output(logits, group_name=group_name, reduction="mean")
            equi0_output = get_output(logits, group_name=group_name, reduction="max")
            output = equitune_output + (equi0_output - equitune_output).detach()
        else:
            output = logits

        ## backprop
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
        # zero the parameter gradients - do it here to save a bit of VRAM before the next iteration
        optimizer.zero_grad()

    return model