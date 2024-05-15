import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl


class WeightNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model_name == "RN50":
            self.fc1 = nn.Linear(1024, 100)
        else:
            self.fc1 = nn.Linear(512, 100)  # thats for clip
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)  # just makes an integer
        self.dp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # check the impact of the gaussian noise even on CNNs
        # x = x + torch.randn(size=x.shape, device=x.device) # add Gaussian to avoid overfitting
        x = self.dp1(self.relu(self.fc1(x)))
        x = self.dp1(self.fc2(x))
        x = self.fc3(x)
        # x = torch.abs(x) + 0.5
        # x = torch.exp(-0.0 * x)
        # x = torch.exp(-0.1 * self.k * x)
        return x  # dim [B, 1]


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
        self.k = nn.Sequential(
            nn.Linear(self.attn_in, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        ).type(self.dtype)

        # todo: second attention layer

    def forward(self, x):
        """

        Args:
            x: features of shape [B, N, D], where N is cardinality of the group

        Returns:

        """
        x = x.type(self.dtype)
        original_shape = x.shape

        values = x.clone().flatten(start_dim=2)

        # preprocess the queries and keys
        x = x.reshape(-1, 1, self.in_dims[1], self.in_dims[2])
        queries = self.per_channel_query_preprocessing(x)
        keys = self.per_channel_key_preprocessing(x)

        queries = self.pre_query_projection(queries.flatten(start_dim=1))
        keys = self.pre_key_projection(keys.flatten(start_dim=1))

        queries = queries.view(original_shape[0], original_shape[1], self.attn_in)
        keys = keys.view(original_shape[0], original_shape[1], self.attn_in)

        queries = self.q(queries)
        keys = self.k(keys)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dim, dtype=torch.float32))

        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)  # dim [B, N, D]
        output = output.mean(dim=1)
        return output.view(original_shape[0], *original_shape[-3:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--model_name', default="RN50", type=str)
    args = parser.parse_args()
    x = torch.randn(size=(4, 3, 224, 224))

    net = WeightNet(args)

    out = net(x)
    print(f"out.shape: {out.shape}")
