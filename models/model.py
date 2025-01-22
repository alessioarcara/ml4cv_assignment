import torch.nn as nn

class ChimeraSeg(nn.Module):
    def __init__(self, backbone, decoder):
        super(ChimeraSeg, self).__init__()
        self.backbone = backbone 
        self.sem_decoder = decoder

    def forward(self, x):
        features = self.backbone(x)
        segmentation_map = self.sem_decoder(features)
        return features, segmentation_map 