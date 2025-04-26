import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/PRBonn/ContMAV/blob/master/src/utils.py
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
                # ew_l1 = (ew_l1 * ew_l1) / (self.var[label] + 1e-8)
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


# https://github.com/Brilhador/tgrs2023/blob/main/utils/losses.py
class PrototypicalGlobalLocalTripletLoss(nn.Module):
    def __init__(self, num_classes, margin_global, margin_local, magnitude=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.magnitude = magnitude
        self.build_anchors()
        distance_function = nn.PairwiseDistance(p=2)
        self.pdist = distance_function.cuda()
        self.triplet_loss_global = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_global
        )
        self.triplet_loss_local = nn.TripletMarginWithDistanceLoss(
            distance_function=self.pdist, swap=False, margin=margin_local
        )

    def build_anchors(self):
        self.anchors = torch.zeros((self.num_classes, self.num_classes), device="cuda")
        self.magnitude = self.magnitude
        for i in range(self.num_classes):  # num_classes
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
        triplet_loss_local = torch.tensor(0.0, device="cuda")
        total_local = 0
        for b in range(shape[0]):
            for c in range(self.num_classes):
                # --
                # positives (TP) (Easy local triplets positives)
                anchors_index = torch.where(
                    (x_pred_local[b] == c) & (y_embeddings_local[b] == c)
                )[0]
                anchors = torch.index_select(
                    x_embeddings_local[b], dim=0, index=anchors_index
                )

                if anchors_index.size()[0] == 0:
                    continue

                # --
                # positives (FN) (False negatives are hard local triplets positives)
                positive_index = torch.where(
                    (x_pred_local[b] != c) & (y_embeddings_local[b] == c)
                )[0]
                positives = torch.index_select(
                    x_embeddings_local[b], dim=0, index=positive_index
                )

                # TN + FP
                negative_index = torch.where(y_embeddings_local[b] != c)[0]
                negatives = torch.index_select(
                    x_embeddings_local[b], dim=0, index=negative_index
                )

                # if no pixels of class x are found, do not compute
                if positive_index.size()[0] == 0 or negative_index.size()[0] == 0:
                    continue

                # random samples
                anchors = anchors[torch.randperm(anchors.size()[0])]
                positives = positives[torch.randperm(positives.size()[0])]
                negatives = negatives[torch.randperm(negatives.size()[0])]

                # slice triplets
                max_triplets = min(
                    [anchors.size()[0], negatives.size()[0], positives.size()[0]]
                )

                anchors = anchors[0:max_triplets]
                negatives = negatives[0:max_triplets]
                positives = positives[0:max_triplets]

                # compute local prototype triplet
                triplet_loss_local += self.triplet_loss_local(
                    anchors, positives, negatives
                )
                total_local += 1

        # avg all triplets loss of images
        triplet_loss_local = triplet_loss_local / total_local

        ###################################
        # GLOBAL ATTRACTION AND REPULSION #
        ###################################

        x_pred_global = x_pred.view(shape[0] * shape[1] * shape[2])
        x_embeddings_global = x_embeddings.view(
            shape[0] * shape[1] * shape[2], shape[3]
        )
        y_embeddings_global = y_embeddings.view(shape[0] * shape[1] * shape[2])

        # loop anchors
        triplet_loss_global = torch.tensor(0.0, device="cuda")
        total_global = 0
        for c, k in enumerate(self.anchors):
            anchor = k.unsqueeze(0)

            # positives (TP) (Easy positives)
            positive_index = torch.where(
                (x_pred_global == c) & (y_embeddings_global == c)
            )[0]
            positives = torch.index_select(
                x_embeddings_global, dim=0, index=positive_index
            )

            # negative (FP) (Hard negatives)
            negative_index = torch.where(
                (x_pred_global == c) & (y_embeddings_global != c)
            )[0]
            negatives = torch.index_select(
                x_embeddings_global, dim=0, index=negative_index
            )

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

            triplet_loss_global += self.triplet_loss_global(
                anchors, positives, negatives
            )
            total_global += 1

        triplet_loss_global = triplet_loss_global / total_global

        return triplet_loss_global + triplet_loss_local
        # return triplet_loss_global.to(dtype=torch.float64) + triplet_loss_local.to(dtype=torch.float64)


class ObjectosphereLoss(nn.Module):
    def __init__(self, xi: float = 1.0, unknown_label: int = -1):
        super(ObjectosphereLoss, self).__init__()
        self.xi = xi
        self.unknown_label = unknown_label

    def forward(
        self,
        feats: torch.Tensor,  # [B, C, H, W]
        labels: torch.Tensor,  # [B, H, W]  (long / int)
    ):
        norm = torch.linalg.norm(feats, ord=2, dim=1)

        unknown = labels == self.unknown_label
        known = ~unknown

        losses = torch.zeros_like(norm, dtype=norm.dtype)
        losses[known] = torch.relu(self.xi - norm[known]).pow(2)
        losses[unknown] = norm[unknown].pow(2)

        return losses.mean()
