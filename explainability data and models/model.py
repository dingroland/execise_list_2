from functools import partial
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import convnext_small, ConvNeXt_Small_Weights, resnet18, ResNet18_Weights


class LayerNorm2d(nn.LayerNorm):
    """2D Layer Normalization module."""

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to a 4D input tensor.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Normalized tensor of the same shape
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNextClassifier(nn.Module):
    """ConvNeXt-based classifier."""

    def __init__(self, output_size: int):
        """
        Initialize the ConvNextClassifier.

        Args:
            output_size (int): Number of classes for classification
        """
        super().__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1).features

        # Uncomment to freeze backbone
        # for parameter in self.backbone.parameters():
        #     parameter.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            norm_layer(768),
            nn.Flatten(1),
            nn.Linear(768, output_size)
        )
        self.transforms = ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Output tensor of shape (N, output_size)
        """
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class ConvResNet(nn.Module):
    """ResNet18-based classifier."""

    def __init__(self, num_classes: int = 200):
        """
        Initialize the ConvResNet.

        Args:
            num_classes (int): Number of classes for classification
        """
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Output tensor of shape (N, num_classes)
        """
        return self.resnet(x)