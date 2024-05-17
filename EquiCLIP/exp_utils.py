import os
import random
import torch
import numpy as np

from torch import cos, sin


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_transform_images(images, group_name="rot90"):
    # group_size is the primary axis!!!
    if group_name == "":
        return torch.stack([images])
    elif group_name == "rot90":
        group_transformed_images = []
        for i in range(4):
            g_images = torch.rot90(images, k=i, dims=(-2, -1))
            group_transformed_images.append(g_images)
        group_transformed_images = torch.stack(group_transformed_images, dim=0)
        return group_transformed_images
    elif group_name == "flip":
        group_transformed_images = []
        for i in range(2):
            if i == 0:
                g_images = images
            else:
                g_images = torch.flip(images, dims=(-2, -1))
            group_transformed_images.append(g_images)
        group_transformed_images = torch.stack(group_transformed_images, dim=0)
        return group_transformed_images
    else:
        raise NotImplementedError


def inverse_transform_images(images, group_name="rot90"):
    if group_name == "":
        return images
    elif group_name == "rot90":
        # expects [B, 4, C, H, W]
        assert len(images.shape) == 5
        assert images.shape[1] == 4, f"images.shape: {images.shape}"
        for i in range(4):
            images[:, i, :, :, :].rot90(k=-i, dims=(-2, -1))
            # i have no idea why but i get the same results when there is a bug in this code
            # images[i].rot90(k=-i, dims=(-2, -1)) # this is wrong because this rotates part of the batch
        return images  # [B, 4, C, H, W]
    else:
        raise NotImplementedError


def verify_invariance(image_features, group_name="rot90", group_transform=None):
    """
    Verify that features of [i] and [i + j * primary_batch_size] are the same
    Args:
        image_features: image features of shape [G*B, C, H, H]
        group_name:
        group_transform: If provided, verifies equivariance of inputs using that transformation function,
            instead of verifying invariance.
            The function should take the tensor to transform and an index of the transformation.
            0 is no transformation.

    Returns:

    """
    if group_name != "rot90":
        raise NotImplementedError

    group_size = 4
    batch_size = image_features.shape[0]
    primary_batch = batch_size // group_size
    for i in range(1, group_size):
        for j in range(primary_batch):
            idx1 = j
            idx2 = i * primary_batch + j
            if group_transform is not None:
                tensor2 = group_transform(image_features[idx2], i)
            else:
                tensor2 = image_features[idx2]
            assert torch.allclose(image_features[idx1], tensor2, rtol=1e-3), \
                (f"idx1: {idx1}, idx2: {idx2},\n"
                 f"tensor1: {image_features[idx1]},\n\ntensor2: {tensor2}")
    print(f"Successfully verified {'invariance' if group_transform is None else 'equivariance'}!")


def verify_weight_equivariance(weights, group_name="rot90"):
    def inverse_weights_transform(w, idx):
        if group_name == "rot90":
            return torch.cat((w[-idx:], w[:-idx]))
        else:
            raise NotImplementedError
    verify_invariance(weights, group_name=group_name, group_transform=inverse_weights_transform)


class RandomRot90(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 4)
        out = torch.rot90(sample, k=k, dims=[self.dim1, self.dim2])
        return out


class RandomFlip(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 2)
        if k == 1:
            out = torch.flip(sample, dims=(self.dim1, self.dim2))
        else:
            out = sample
        return out


random_rot90 = RandomRot90()
random_flip = RandomFlip()


def random_transformed_images(x, data_transformations=""):
    if data_transformations == "rot90":
        x = random_rot90(x)
    elif data_transformations == "flip":
        x = random_flip(x)
    else:
        x = x
    return x
