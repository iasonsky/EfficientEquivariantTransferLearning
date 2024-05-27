import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl


class WeightNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model_name != "RN50":
            raise NotImplementedError

        self.per_channel_out = 8
        self.per_channel_channels = 32
        self.dtype = torch.float32

        self.per_channel_preprocessing = nn.Sequential(
            nn.Conv2d(1, self.per_channel_channels, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(self.per_channel_channels, self.per_channel_channels, kernel_size=3),
            # nn.ReLU(),
        ).type(self.dtype)

        self.pre_projection = nn.Sequential(
            nn.Linear(self.per_channel_channels * 5 * 5, self.per_channel_out),
            nn.ReLU(),
        ).type(self.dtype)

        self.full_in = 2048 * self.per_channel_out
        self.hidden = 128

        self.main = nn.Sequential(
            nn.Linear(self.full_in, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        ).type(self.dtype)

    def forward(self, x):
        # takes [B, G, *feature_dims]
        x = x.type(self.dtype)
        original_shape = x.shape

        # put the n_channels (2048) in the batch dim
        x = x.reshape(-1, 1, original_shape[-2], original_shape[-1])
        x = self.per_channel_preprocessing(x)
        x = self.pre_projection(x.flatten(start_dim=1))

        x = x.view(original_shape[0] * original_shape[1], self.full_in)
        x = self.main(x)
        return x  # dim [B*G, 1]


class AttentionAggregation(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name != "RN50":
            raise NotImplementedError
        self.in_dims = [2048, 7, 7]
        self.dim = 128

        self.per_channel_out = 8
        self.per_channel_channels = 32
        self.dtype = torch.float32

        # the task of this thing is to encode the general "direction" of features
        self.per_channel_query_preprocessing = nn.Sequential(
            nn.Conv2d(1, self.per_channel_channels, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(self.per_channel_channels, self.per_channel_channels, kernel_size=3),
            # nn.ReLU(),
        ).type(self.dtype)

        self.pre_query_projection = nn.Sequential(
            nn.Linear(self.per_channel_channels * 5 * 5, self.per_channel_out),
            nn.ReLU()
        ).type(self.dtype)

        self.per_channel_key_preprocessing = nn.Sequential(
            nn.Conv2d(1, self.per_channel_channels, kernel_size=3),
            nn.ReLU(),
            # nn.Conv2d(self.per_channel_channels, self.per_channel_channels, kernel_size=3),
            # nn.ReLU(),
        ).type(self.dtype)

        self.pre_key_projection = nn.Sequential(
            nn.Linear(self.per_channel_channels * 5 * 5, self.per_channel_out),
            nn.ReLU()
        ).type(self.dtype)

        self.attn_in = 2048 * self.per_channel_out

        self.q = nn.Sequential(
            nn.Linear(self.attn_in, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        ).type(self.dtype)
        nn.init.kaiming_uniform_(self.q[0].weight, mode='fan_in', nonlinearity='relu')
        self.k = nn.Sequential(
            nn.Linear(self.attn_in, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        ).type(self.dtype)
        nn.init.kaiming_uniform_(self.k[0].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        """
        This function only computes attention weights

        Args:
            x: features of shape [B, N, D], where N is cardinality of the group

        Returns:

        """
        x = x.type(self.dtype)
        original_shape = x.shape

        # preprocess the queries and keys
        x = x.reshape(-1, 1, self.in_dims[1], self.in_dims[2])
        queries = self.per_channel_query_preprocessing(x)
        keys = self.per_channel_key_preprocessing(x)

        queries = self.pre_query_projection(queries.flatten(start_dim=1))
        keys = self.pre_key_projection(keys.flatten(start_dim=1))

        # to B, N, D_in
        queries = queries.view(original_shape[0], original_shape[1], self.attn_in)
        keys = keys.view(original_shape[0], original_shape[1], self.attn_in)

        queries = self.q(queries)
        keys = self.k(keys)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dim, dtype=torch.float32))

        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--model_name', default="RN50", type=str)
    args = parser.parse_args()
    x = torch.randn(size=(4, 3, 224, 224))

    net = WeightNet(args)

    out = net(x)
    print(f"out.shape: {out.shape}")
