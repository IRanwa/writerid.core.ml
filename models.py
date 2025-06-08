import torch
from torch import nn
from torch import Tensor

class PrototypicalNetworkModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images: Tensor, support_labels: Tensor, query_images: Tensor) -> Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        support_labels = support_labels.to(z_support.device)
        unique_labels = torch.unique(support_labels)
        n_way = len(unique_labels)
        all_prototypes = []
        for label_idx in range(n_way):
            current_label_val = unique_labels[label_idx]
            label_mask = torch.nonzero(support_labels == current_label_val, as_tuple=False)
            if label_mask.numel() > 0:
                label_mask = label_mask.squeeze(-1)
                proto = z_support[label_mask].mean(0)
                all_prototypes.append(proto)
        z_proto = torch.stack(all_prototypes)
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores