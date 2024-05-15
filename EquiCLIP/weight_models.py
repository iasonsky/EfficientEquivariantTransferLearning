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
        self.dim = 50
        self.q = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 50),
            nn.ReLU(),
            nn.Linear(50, self.dim)
        ).half()
        self.k = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 50),
            nn.ReLU(),
            nn.Linear(50, self.dim)
        ).half()

    def forward(self, x):
        """

        Args:
            x: features of shape [B, N, D], where N is cardinality of the group

        Returns:

        """
        original_shape = x.shape
        queries = self.q(x.flatten(start_dim=2))
        keys = self.k(x.flatten(start_dim=2))
        values = x.clone().flatten(start_dim=2)

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
