import torch
from torch import nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    googlenet, GoogLeNet_Weights
)

class BackboneNetworkHandler:
    def __init__(self, name: str = "googlenet", pretrained: bool = True):
        self.name = name.lower()
        self.pretrained = pretrained
        self.model: nn.Module
        self.target_channels: int = 1
        self._create_backbone()

    def _adapt_to_single_channel(self, model: nn.Module):
        layer_path_map = {
            "googlenet": "conv1.conv",
            "resnet18": "conv1"
        }
        path = layer_path_map.get(self.name)
        path_parts = path.split('.')
        parent_module = model
        for part in path_parts[:-1]:
            parent_module = getattr(parent_module, part)
        layer_name = path_parts[-1]
        original_layer = getattr(parent_module, layer_name)
        original_weights = original_layer.weight
        new_layer = nn.Conv2d(
            in_channels=self.target_channels,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            groups=original_layer.groups,
            bias=(original_layer.bias is not None)
        )
        with torch.no_grad():
            new_layer.weight.data = original_weights.data.mean(dim=1, keepdim=True)
            if new_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data
        setattr(parent_module, layer_name, new_layer)

    def _create_backbone(self):
        weights_map = {
            "googlenet": GoogLeNet_Weights.DEFAULT,
            "resnet18": ResNet18_Weights.DEFAULT
        }
        model_fn_map = {
            "googlenet": googlenet,
            "resnet18": resnet18
        }
        weights = weights_map[self.name] if self.pretrained else None
        base_model = model_fn_map[self.name](weights=weights)
        
        if self.pretrained:
            self._adapt_to_single_channel(base_model)
            
        if self.name == "googlenet":
            base_model._transform_input = lambda x: x
            base_model.fc = nn.Identity()
            self.model = base_model
        elif self.name == "resnet18":
            base_model.fc = nn.Identity()
            self.model = base_model

    def get_model(self) -> nn.Module:
        return self.model