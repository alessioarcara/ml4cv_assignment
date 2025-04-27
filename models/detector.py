import torch
import torch.nn as nn
from tqdm.notebook import tqdm


class RunningCenters(nn.Module):
    def __init__(self, n_classes: int):
        super(RunningCenters, self).__init__()
        self.num_classes = n_classes
        self.register_buffer(
            "running_centers", torch.zeros(self.num_classes, self.num_classes)
        )
        self.register_buffer("running_counts", torch.zeros(self.num_classes))

    @property
    def centers(self) -> torch.Tensor:
        return self.running_centers / self.running_counts.clamp(min=1).unsqueeze(1)

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        for cls in range(self.num_classes):
            class_mask = labels == cls
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                class_sum = class_embeddings.sum(dim=0)
                self.running_centers[cls] += class_sum
                self.running_counts[cls] += class_mask.sum().float()

    def fit(self, model, train_loader, device):
        self.to(device)
        model = model.to(device).eval()

        self.reset()

        with torch.no_grad():
            for imgs, masks in tqdm(train_loader, desc="Computing class centers"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).view(-1)

                logits = model(imgs)  # [B, C, H, W]
                embeds = logits.permute(0, 2, 3, 1).reshape(
                    -1, self.num_classes
                )  # [B*H*W, C]

                self.update(embeds, masks)

        return self

    def reset(self):
        self.running_centers.zero_()
        self.running_counts.zero_()
