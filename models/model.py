import torch.nn as nn

class ChimeraSeg(nn.Module):
    def __init__(self, encoder, decoder):
        super(ChimeraSeg, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        prelogits, logits = self.decoder(features)
        return prelogits, logits 