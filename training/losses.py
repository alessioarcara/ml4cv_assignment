import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassDescriptorLoss(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        # Current epoch statistics
        self.register_buffer('curr_means', torch.zeros(K, D))
        self.register_buffer('curr_vars', torch.ones(K, D))
        self.register_buffer('counts', torch.zeros(K))
        # Previous epoch statistics (e-1)
        self.register_buffer('prev_means', None)
        self.register_buffer('prev_vars', None)

    @torch.no_grad()
    def update_class_statistics(self, features, pred_classes, true_classes):
        for k in range(self.K):
            # Ωk: set of pixels whose predicted and ground truth labels are both k
            class_mask = true_classes == k
            pred_mask = pred_classes == k
            tps_mask = torch.logical_and(class_mask, pred_mask)
            n_pixels = tps_mask.sum()

            if n_pixels > 0:
                class_features = features[tps_mask]               # [N, D] where N is number of true positives
                avg_features = class_features.mean(0)  # [D]
                self.curr_means[k] = ((self.curr_means[k] * self.counts[k] + avg_features * n_pixels) / 
                        (self.counts[k] + n_pixels))
                self.counts[k] = self.counts[k] + n_pixels

                diff = class_features - self.curr_means[k]    # [N, D]
                self.curr_vars[k] = (diff * diff).mean(0)     # [D] 
                
    def compute_image_loss(self, features, true_classes):
        if self.prev_means is None:
            return torch.tensor(0.0, device=features.device)
        
        total_loss = 0.0

        for k in range(self.K):
            # Ωk: pixels with ground truth label k
            class_mask = (true_classes == k)

            if class_mask.sum() > 0:
                class_features = features[class_mask]                    # [N, D]
                prev_mean = self.prev_means[k]                           # [D]
                prev_var = self.prev_vars[k]                             # [D]

                l1_distances = torch.abs(class_features - prev_mean) # [N, D]
                normalized_distances = l1_distances / (prev_var + 1e-8)  # [N]
                total_loss += normalized_distances.mean()
        
        return total_loss
    
    def on_epoch_end(self):
        self.prev_means = self.curr_means.clone()
        self.prev_vars = self.curr_vars.clone()
        self.curr_means = torch.zeros((self.K, self.D))
        self.curr_vars = torch.zeros((self.K, self.D))
        self.counts.zero_()

    def forward(self, logits, true_classes, features, is_train):
        """
        Args:
            logits: [B, C, H, W] tensor of logits
            targets: [B, H, W] tensor of class labels
            features: [B, D, H, W] tensor of pre-logit features
        """
        device = features.device
        self.to(device)

        B, D, H, W = features.shape
        batch_loss = 0.0

        for b in range(B):
            img_features = features[b].permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
            img_true = true_classes[b].reshape(-1)                      # [H*W]
            img_pred = torch.argmax(logits[b], dim=0).reshape(-1)       # [H*W]

            if is_train:
                self.update_class_statistics(img_features, img_pred, img_true)
            img_loss = self.compute_image_loss(img_features, img_true)
            batch_loss += img_loss

        return batch_loss / B
    

class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.delta = delta
        self.count = torch.zeros(self.n_classes).cuda()  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        self.var = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.permute(0, 2, 3, 1)
        for label in gt_labels:
            if label == 255:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() == 0:
                continue
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            # features is running mean for mav
            self.features[label] = (
                self.features[label] * self.count[label] + avg_mav * n_tps
            )

            self.ex[label] += (logits_tps).sum(dim=0)
            self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(
        self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: bool = False 
    ) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features == None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        logits_permuted = logits.permute(0, 2, 3, 1)

        acc_loss = torch.tensor(0.0).cuda()
        for label in gt_labels[:-1]:
            mav = self.previous_features[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                ew_l1 = self.criterion(logs, mav)
                #ew_l1 = (ew_l1 * ew_l1) / (self.var[label] + 1e-8)
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                acc_loss += ew_l1.mean()

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for c in self.var.keys():
            self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (
                self.count[c] + 1e-8
            )

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes)  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor


class PrototypicalGlobalLocalTripletLoss(nn.Module):
    
    def __init__(self, num_classes, margin_global, margin_local, magnitude=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.magnitude = magnitude
        self.build_anchors()
        distance_function = nn.PairwiseDistance(p=2)
        self.pdist = distance_function.cuda()
        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist,
            swap=False,
            margin = margin_global
            )
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist,
            swap=False,
            margin = margin_local
            )
    
    def build_anchors(self):
        self.anchors = torch.zeros((self.num_classes, self.num_classes), device='cuda')
        self.magnitude = self.magnitude
        for i in range(self.num_classes): # num_classes
            self.anchors[i][i] = self.magnitude
       
    def forward(self, x_embeddings, y_embeddings):
        
        # reshape
        x_embeddings = x_embeddings.permute(0, 2, 3, 1).contiguous()
        shape = x_embeddings.size()
        
        # compute argmax
        x_pred = torch.argmax(x_embeddings, dim=3)
        
        ###################################
        # LOCAL ATTRACTION AND REPULSION #
        ###################################
        
        x_pred_local = x_pred.view(shape[0], shape[1] * shape[2])
        x_embeddings_local = x_embeddings.view(shape[0], shape[1] * shape[2], shape[3]) 
        y_embeddings_local = y_embeddings.view(shape[0], shape[1] * shape[2])
        
        # loop images in batch (to compute local attraction and repulsion)
        triplet_loss_local = torch.tensor(0., device='cuda')
        total_local = 0
        for b in range(shape[0]):
            for c in range(self.num_classes):
            
                # --
                # positives (TP) (Easy local triplets positives)
                anchors_index = torch.where((x_pred_local[b] == c) & (y_embeddings_local[b] == c))[0]
                anchors = torch.index_select(x_embeddings_local[b], dim=0, index=anchors_index)               

                if anchors_index.size()[0] == 0:
                    continue
                
                # --
                # positives (FN) (False negatives are hard local triplets positives)
                positive_index = torch.where((x_pred_local[b] != c) & (y_embeddings_local[b] == c))[0]
                positives = torch.index_select(x_embeddings_local[b], dim=0, index=positive_index) 

                # TN + FP
                negative_index = torch.where(y_embeddings_local[b] != c)[0] 
                negatives = torch.index_select(x_embeddings_local[b], dim=0, index=negative_index) 

                # if no pixels of class x are found, do not compute
                if positive_index.size()[0] == 0 or negative_index.size()[0] == 0:
                    continue
                
                # random samples
                anchors = anchors[torch.randperm(anchors.size()[0])]
                positives = positives[torch.randperm(positives.size()[0])]
                negatives = negatives[torch.randperm(negatives.size()[0])]
                
                # slice triplets
                max_triplets = min([anchors.size()[0], negatives.size()[0], positives.size()[0]])
                
                anchors = anchors[0:max_triplets]
                negatives = negatives[0:max_triplets]
                positives = positives[0:max_triplets]
                
                # compute local prototype triplet
                triplet_loss_local += self.triplet_loss_local(anchors, positives, negatives) 
                total_local += 1
                    
        # avg all triplets loss of images
        triplet_loss_local = triplet_loss_local / total_local
                
        ###################################
        # GLOBAL ATTRACTION AND REPULSION #
        ###################################
        
        x_pred_global = x_pred.view(shape[0] * shape[1] * shape[2])
        x_embeddings_global = x_embeddings.view(shape[0] * shape[1] * shape[2], shape[3]) 
        y_embeddings_global = y_embeddings.view(shape[0] * shape[1] * shape[2])
        
        # loop anchors 
        triplet_loss_global = torch.tensor(0., device='cuda')
        total_global = 0
        for c, k in enumerate(self.anchors):

            anchor = k.unsqueeze(0)
            
            # positives (TP) (Easy positives)
            positive_index = torch.where((x_pred_global == c) & (y_embeddings_global == c))[0]
            positives = torch.index_select(x_embeddings_global, dim=0, index=positive_index)
            
            # negative (FP) (Hard negatives)
            negative_index = torch.where((x_pred_global == c) & (y_embeddings_global != c))[0]
            negatives = torch.index_select(x_embeddings_global, dim=0, index=negative_index)
            
            # if no pixels of class x are found, do not compute
            if positive_index.size()[0] == 0 or negative_index.size()[0] == 0:
                continue
            
            # random samples
            positives = positives[torch.randperm(positives.size()[0])]
            negatives = negatives[torch.randperm(negatives.size()[0])]
            
            # slice triplets
            max_triplets = min([negatives.size()[0], positives.size()[0]])
            
            negatives = negatives[0:max_triplets]
            positives = positives[0:max_triplets]

            anchors = anchor.expand(max_triplets, -1)
            
            triplet_loss_global += self.triplet_loss_global(anchors, positives, negatives) 
            total_global += 1
            
        triplet_loss_global = triplet_loss_global / total_global   
        
        return triplet_loss_global + triplet_loss_local
        #return triplet_loss_global.to(dtype=torch.float64) + triplet_loss_local.to(dtype=torch.float64)


if __name__ == "__main__":
    B, D, H, W = 16, 128, 256, 256
    K = 10
    features = torch.randn(B, D, H, W)
    logits = torch.randn(B, K, H, W)
    true_masks = torch.randint(0, K, (B, H, W))
    
    l_feat = ClassDescriptorLoss(K, D)
    print(l_feat.forward(logits, true_masks, features, True))
    l_feat.on_epoch_end()
    print(l_feat.forward(logits, true_masks, features, True))