import logging
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F

from tqdm.autonotebook import tqdm

from weighted_equitune_utils import compute_logits
from exp_utils import group_transform_images, random_transformed_images
from weighted_equitune_utils import get_output

group_sizes = {"rot90": 4., "flip": 2., "": 1.}


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def compute_precision(output, target):
    pred = output.argmax(dim=1).cpu().numpy()  # Get the index of the max log-probability
    target = target.cpu().numpy()
    precision = precision_score(target, pred, average='macro', zero_division=0.0)  # Use 'macro' for multi-class
    return precision


def compute_recall(output, target):
    pred = output.argmax(dim=1).cpu().numpy()  # Get the index of the max log-probability
    target = target.cpu().numpy()
    recall = recall_score(target, pred, average='macro', zero_division=0.0)  # Use 'macro' for multi-class
    return recall


def compute_f1(output, target):
    pred = output.argmax(dim=1).cpu().numpy()  # Get the index of the max log-probability
    target = target.cpu().numpy()
    f1 = f1_score(target, pred, average='macro')  # Use 'macro' for multi-class
    return f1


def eval_clip(args, model, zeroshot_weights, loader, data_transformations="", group_name="", device="cuda:0",
              feature_combination_module=None,
              val=False, save_scores=False):
    with torch.no_grad():
        top1, top5, n_samples = 0., 0., 0
        precision_total, recall_total, f1_total = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader, desc="Evaluating CLIP")):
            if val and i == 50:
                break
            images = images.to(device)  # dim [batch_size, c_in, H, H]

            if args.validate_equivariance:
                # for verification apply group transformations a predictable manner, duplicating the batch
                original_shape = images.shape
                images = group_transform_images(images, group_name=group_name)
                images = images.view(-1, *original_shape[1:])
                # now we have somthing like [up1, up2, right1, right2, down1, down2, left1, left2]
            else:
                images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

            # images = torch.rot90(images, k=1, dims=(-2, -1))
            group_images = group_transform_images(images,
                                                  group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
            group_images_shape = group_images.shape

            # dim [group_size * batch_size, c_in, H, H]
            group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                                group_images_shape[3], group_images_shape[3])
            # print(f"images.shape: {images.shape}")
            target = target.to(device)
            # print(f"target.shape: {target.shape}")

            logits = compute_logits(args, model, feature_combination_module, group_images,
                                    zeroshot_weights, group_name, validate_equivariance=args.validate_equivariance,
                                    log_variance=False)

            if args.validate_equivariance:
                target = target.repeat(int(group_sizes[group_name]))

            if args.method == "equizero":
                output = get_output(logits, group_name=group_name, reduction="max")
            else:
                output = logits

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            precision = compute_precision(output, target)
            recall = compute_recall(output, target)
            f1 = compute_f1(output, target)

            top1 += acc1
            top5 += acc5
            precision_total += precision
            recall_total += recall
            f1_total += f1
            n_samples += images.size(0)

    top1 = (top1 / n_samples) * 100
    top5 = (top5 / n_samples) * 100
    total_batches = 50 if val else len(loader)
    precision_avg = (precision_total / total_batches) * 100
    recall_avg = (recall_total / total_batches) * 100
    f1_avg = (f1_total / total_batches) * 100

    info = [
        f"Dataset: {args.dataset_name}",
        f"Model: {args.model_name}",
        f"Method: {args.method}",
        f"Group: {args.group_name}",
        f"Data transformation: {args.data_transformations}",
        f"Top-1 accuracy: {top1:.2f}",
        f"Top-5 accuracy: {top5:.2f}",
        f"Precision: {precision_avg:.2f}",
        f"Recall: {recall_avg:.2f}",
        f"F1 score: {f1_avg:.2f}",
    ]

    for message in info:
        logging.info(message)
        print(message)

    # Save the top-1 accuracy in a folder 
    if save_scores:
        folder = f"results/{args.dataset_name}/{args.model_name}/{args.method}/{args.data_transformations}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/top1_accuracy.txt", "w") as f:
            f.write(f"{top1:.2f}")
        # save the top-5 accuracy as well
        with open(f"{folder}/top5_accuracy.txt", "w") as f:
            f.write(f"{top5:.2f}")

    return top1, top5, precision_avg, recall_avg, f1_avg
