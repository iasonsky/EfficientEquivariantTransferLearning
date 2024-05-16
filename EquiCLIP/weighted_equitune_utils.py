from typing import Optional, Union

import torch
import torch.nn.functional as F
from clip.model import CLIP, ModifiedResNet

from tqdm import tqdm

from weight_models import AttentionAggregation, WeightNet
from exp_utils import group_transform_images, random_transformed_images, inverse_transform_images

group_sizes = {"rot90": 4., "flip": 2., "": 1.}

def cycle(iterable):
    # this does not reset the iterable,
    # so it hangs the process with an infinite loop when iterable reaches the end (i think)
    while True:
        for x in iterable:
            yield x


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_equi0_output(output, target, topk=(1,), group_name=""):
    if group_name == "":
      return output
    elif group_name == "rot90":
      group_size = 4
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      return output
    elif group_name == "flip":
      group_size = 2
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      return output
    else:
      raise NotImplementedError


def get_equitune_output(output, target, topk=(1,), group_name=""):
    if group_name == "":
        pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
        return output
    elif group_name=="rot90":
        group_size = 4
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        return output
    elif group_name == "flip":
        group_size = 2
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        return output
    else:
        raise NotImplementedError


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
):
    if args.method == "attention" or args.method == "equitune":
        image_features = conv_forward(model.visual, group_images.type(model.dtype))  # dim [group_size * batch_size, *feat_dims]
        # feature dims are [2048, 7, 7] for RN50

        # unrotate
        # image_features = inverse_transform_images(image_features, group_name=group_name)  # [group_size, B, C, H, H]

        # to B, N, D form. where D is [C, H, H], and N is the group size
        image_features = image_features.transpose(0, 1)

        if args.method == "attention":
            raise NotImplementedError("no inverse transform is applied. should implement like discussed with Milan.")
            combined_features = feature_combination_module(image_features)  # dim [batch_size, *feat_size]
        else:
            weights = feature_combination_module(image_features)  # dim [batch_size * group_size, 1]
            weights = weights.reshape(-1, int(group_sizes[group_name]))  # dim [batch_size, group_size]
            # this softmax normalizes the weights for each group
            weights = F.softmax(weights, dim=-1)  # dim [batch_size, group_size]

            weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4).type(image_features.dtype)

            # unrotate after the weights have been calculated
            image_features = inverse_transform_images(image_features, group_name=group_name)  # [group_size, B, C, H, H]

            # to verify: is this truly equivariant?

            # sum and not mean because they normalized anyway
            combined_features = torch.sum(image_features * weights, dim=1)  # dim [batch_size, *feat_dims]
        # take the avg
        # combined_features = image_features.mean(dim=1)  # dim [batch_size, **feat_dims]

        # we now have EQUIVARIANT features (correction: they are not actually)

        final_features = finish_resnet_forward(model.visual, combined_features.type(model.dtype))

        logits = final_features @ zeroshot_weights  # B, num_classes
    elif args.method == "vanilla" or feature_combination_module is None:
        image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
        logits = args.logit_factor * image_features @ zeroshot_weights
        if args.softmax:
            logits = torch.nn.functional.softmax(logits, dim=-1)
    else:
        raise NotImplementedError
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
    training_iterator = cycle(iter(loader))

    for _ in tqdm(range(num_iterations)):
        (images, target) = next(training_iterator)
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
        # if args.method == "equitune":
        #     output = get_equitune_output(logits, target, topk=(1,), group_name=group_name)  # dim [batch_size, num_classes=1000]
        if args.method == "equizero":
            equitune_output = get_equitune_output(logits, target, topk=(1,), group_name=group_name)
            equi0_output = get_equi0_output(logits, target, topk=(1,), group_name=group_name)
            output = equitune_output + (equi0_output - equitune_output).detach()
        elif args.method == "attention" or args.method == "equitune":
            output = logits
        else:
            output = get_equi0_output(logits, target, topk=(1,), group_name="")

        ## backprop
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            # print(lr_scheduler.get_last_lr())
            lr_scheduler.step()

    return model
