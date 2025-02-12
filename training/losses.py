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

    @torch.no_grad()
    def update_class_statistics(self, features, pred_classes, true_classes):
        for k in range(self.K):
            true_positives = (pred_classes == k) & (true_classes == k) # true positives for class k in the image
            if true_positives.sum() > 0:
                class_features = features[true_positives]   # [N, D] where N is number of true positives
                self.curr_means[k] = class_features.mean(0) # [D]
                diff = class_features - self.curr_means[k]  # [N, D]
                self.curr_vars[k] = (diff * diff).mean(0)   # [D] 
                
    def compute_image_loss(self, features, true_classes):
        if self.prev_means is None:
            return torch.tensor(0.0, device=features.device)
        
        total_loss = 0.0

        for k in range(self.K):
            class_mask = (true_classes == k)
            if class_mask.sum() > 0:
                class_features = features[class_mask]               # [N, D]
                mean = self.prev_means[k]                           # [D]
                var = self.prev_vars[k] + 1e-6                      # [D]

                diff = class_features - mean                        # [N, D]
                normalized_diff = diff / torch.sqrt(var)            # [N, D]
                l2_dist_squared = (normalized_diff ** 2).sum(dim=1) # [N]

                total_loss += l2_dist_squared.sum()
        
        return total_loss
    
    def on_epoch_end(self):
        self.prev_means = self.curr_means.clone()
        self.prev_vars = self.curr_vars.clone()

    def forward(self, logits, true_classes, features):
        """

        Args:
            features: [B, D, H, W] tensor of pre-logit features
            logits: [B, C, H, W] tensor of logits
            targets: [B, H, W] tensor of class labels
        """
        device = features.device
        self.curr_means = self.curr_means.to(device)
        self.curr_vars = self.curr_vars.to(device)
        if self.prev_means is not None:
            self.prev_means = self.prev_means.to(device)
            self.prev_vars = self.prev_vars.to(device)

        B, D, H, W = features.shape
        batch_loss = 0.0
        num_pixels_per_image = H * W

        for b in range(B):
            img_features = features[b].permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
            img_true = true_classes[b].reshape(-1)                      # [H*W]
            img_pred = torch.argmax(logits[b], dim=0).reshape(-1)       # [H*W]

            self.update_class_statistics(img_features, img_true, img_pred)
            img_loss = self.compute_image_loss(img_features, img_true)
            batch_loss += img_loss

        return batch_loss / (B * num_pixels_per_image) 


if __name__ == "__main__":
    B, D, H, W = 16, 128, 256, 256
    K = 10

    prelogits = torch.randn (B, D, H, W)
    pred_masks = torch.randint(0 , 2, (B, K, H, W)).float()
    true_masks = torch.randint(0 , K, (B, H, W)).float()

    l_feat = ClassDescriptorLoss(K, D)
    print(l_feat.forward(prelogits, pred_masks, true_masks))
    l_feat.on_epoch_end()
    print(l_feat.forward(prelogits, pred_masks, true_masks))
    pass