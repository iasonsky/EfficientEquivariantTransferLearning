from typing import Optional, Union

import torch
import torch.nn.functional as F
from clip.model import CLIP, ModifiedResNet
import wandb
from tqdm.autonotebook import tqdm, trange
from itertools import islice
from weight_models import AttentionAggregation, WeightNet
from exp_utils import group_transform_images, random_transformed_images, inverse_transform_images, verify_invariance, \
    verify_weight_equivariance

group_sizes = {"rot90": 4., "flip": 2., "": 1.}


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


def conv_forward(resnet: ModifiedResNet, x):
    def stem(x):
        x = resnet.relu1(resnet.bn1(resnet.conv1(x)))
        x = resnet.relu2(resnet.bn2(resnet.conv2(x)))
        x = resnet.relu3(resnet.bn3(resnet.conv3(x)))
        x = resnet.avgpool(x)
        return x

    x = x.type(resnet.conv1.weight.dtype)
    x = stem(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    return x


def finish_resnet_forward(resnet: ModifiedResNet, x):
    x = resnet.attnpool(x)
    return x


def compute_logits(
        args,
        model: CLIP,
        feature_combination_module: Union[WeightNet, AttentionAggregation],
        group_images,
        zeroshot_weights,
        group_name,
        validate_equivariance=False,  # here it is a separate arg because it is only called in validation
        return_weights=False,
        log_variance=True,
):
    lambda_weights = None
    if args.method == "attention" or args.method == "equitune":
        group_size = int(group_sizes[group_name])
        image_features = conv_forward(model.visual, group_images.type(model.dtype))  # dim [group_size * batch_size, *feat_dims]
        # feature dims are [2048, 7, 7] for RN50

        # convert to batch-first representation because i like it
        assert len(image_features.shape) == 4  # for batch+group, channel, height, width
        image_features = image_features.view(4, -1, image_features.shape[-3], image_features.shape[-2], image_features.shape[-1])
        # to B, G, *D form. where D is [C, H, H], and G is the group size
        image_features = image_features.transpose(0, 1)

        if args.method == "attention":
            original_shape = image_features.shape

            attention_weights = feature_combination_module(image_features)  # dim [batch_size, group_size, group_size]
            assert len(attention_weights.shape) == 3 and attention_weights.shape[1] == attention_weights.shape[2]

            if log_variance:
                wandb.log({"weight_variance": torch.var(attention_weights.mean(dim=1)).item()}, commit=False)

            image_features = inverse_transform_images(image_features, group_name=group_name)  # [B, G, C, H, H]
            assert image_features.shape[1] == group_size

            # Finish applying attention
            values = image_features.flatten(start_dim=2).type(feature_combination_module.dtype)  # [B, G, C*H*H]
            combined_features = torch.matmul(attention_weights, values)  # dim [B, N, D]
            combined_features = combined_features.view(original_shape)

            # mean features over the group
            combined_features = combined_features.mean(dim=1)
            # verify it looks like one feature set
            assert combined_features.shape[0] == original_shape[0]
            assert combined_features.shape[-3:] == original_shape[-3:]
            assert len(combined_features.shape) == 4
            if return_weights:
                lambda_weights = attention_weights
        else:
            weights = feature_combination_module(image_features)  # dim [batch_size * group_size, 1]
            weights = weights.reshape(-1, group_size)  # dim [batch_size, group_size]
            # this softmax normalizes the weights for each group
            weights = F.softmax(weights, dim=-1)  # dim [batch_size, group_size]

            if log_variance:
                # log weight variance to see if model diverges from "mean" weights
                wandb.log({"weight_variance": torch.var(weights).item()}, commit=False)

            weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4).type(image_features.dtype)

            # unrotate after the weights have been calculated
            image_features = inverse_transform_images(image_features, group_name=group_name)  # [B, G, C, H, H]
            assert image_features.shape[1] == group_size

            # sum and not mean because they normalized anyway
            combined_features = torch.sum(image_features * weights, dim=1)  # dim [batch_size, *feat_dims]

            if return_weights:
                lambda_weights = weights

        if validate_equivariance:
            # actually verify invariance of combined features because that is much easier
            # than verifying equivariance of the attention weights
            verify_invariance(combined_features, group_name=group_name)
            # for weightnet we can do that though
            if args.method == "equitune":
                weights = weights.reshape(-1, group_size)
                verify_weight_equivariance(weights, group_name=group_name)

        final_features = finish_resnet_forward(model.visual, combined_features.type(model.dtype))

        logits = final_features @ zeroshot_weights  # B, num_classes
    elif args.method == "vanilla" or feature_combination_module is None:
        image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
        logits = args.logit_factor * image_features @ zeroshot_weights
        if args.softmax:
            logits = torch.nn.functional.softmax(logits, dim=-1)
    else:
        raise NotImplementedError

    if return_weights:
        return logits, lambda_weights
    else:
        return logits


def weighted_equitune_clip(
        args, model: CLIP,
        feature_combination_module: Union[WeightNet, AttentionAggregation],
        optimizer, criterion,
        zeroshot_weights, loader,
        data_transformations="", group_name="",
        num_iterations=100, device="cuda:0", lr_scheduler=None
):
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
        device:

    Returns:

    """
    torch.autograd.set_detect_anomaly(True)

    # Create a limited loader to avoid running the entire dataset
    limited_loader = islice(loader, num_iterations)
    # Count the actual number of items in limited_loader
    actual_iterations = min(num_iterations, len(loader))

    for images, target in tqdm(limited_loader, desc="Training CLIP and/or WeightNet", total=actual_iterations):
        images = images.to(device)  # dim [batch_size, c_in, H, H]
        images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

        group_images = group_transform_images(images,
                                              group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
        group_images_shape = group_images.shape

        # dim [group_size * batch_size, c_in, H, H]
        group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                            group_images_shape[3], group_images_shape[3])
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        logits = compute_logits(args, model, feature_combination_module, group_images,
                                zeroshot_weights, group_name)

        # measure accuracy
        if args.method == "equizero":
            equitune_output = get_output(logits, group_name=group_name, reduction="mean")
            equi0_output = get_output(logits, group_name=group_name, reduction="max")
            output = equitune_output + (equi0_output - equitune_output).detach()
        else:
            output = logits

        # backprop
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            # print(lr_scheduler.get_last_lr())
            lr_scheduler.step()

        wandb.log({"loss": loss.item()})
        # zero the parameter gradients - do it here to save a bit of VRAM before the next iteration
        optimizer.zero_grad()

    return model
