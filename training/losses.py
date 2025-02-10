import torch
import torch.nn as nn

class ClassDescriptorLoss(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.register_buffer('curr_means', torch.zeros(K, D))
        self.register_buffer('curr_vars', torch.ones(K, D))
        self.register_buffer('prev_means', None)
        self.register_buffer('prev_vars', None)

    def update_class_statistics(self, features, pred_classes, true_classes):
        for k in range(self.K):
            true_positives = (pred_classes == k) & (true_classes == k) # true positives for class k in the image
            if true_positives.sum() > 0:
                class_features = features[true_positives] # [N, D] where N is number of true positives

                self.curr_means[k] = class_features.mean(0) # [D]
                diff = class_features - self.curr_means[k]  # [N, D]
                self.curr_means[k] = (diff * diff).mean(0)  # [D]

    def forward(self, features, logits, true_classes):
        """

        Args:
            features: [B, D, H, W] tensor of pre-logit features
            logits: [B, C, H, W] tensor of logits
            targets: [B, H, W] tensor of class labels
        """
        B, D, H, W = features.shape

        for b in range(B):
            img_features = features[b].permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
            img_true = true_classes[b].reshape(-1)  # [H*W]
            img_pred = torch.argmax(logits[b], dim=0).reshape(-1)  # [H*W]

            self.update_class_statistics(img_features, img_true, img_pred)

    def compute_image_loss(self, true_classes):
        for k in range(self.K):
            class_mask = (true_classes == k)
            if class_mask.sum() > 0:
                pass


if __name__ == "__main__":
    B, D, H, W = 16, 128, 256, 256
    K = 10

    prelogits = torch.randn (B, D, H, W)
    pred_masks = torch.randint(0 , 2, (B, K, H, W)).float()
    true_masks = torch.randint(0 , K, (B, H, W)).float()

    loss = ClassDescriptorLoss(K, D)
    loss.forward(prelogits, pred_masks, true_masks)
    pass