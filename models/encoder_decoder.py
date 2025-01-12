import torch.nn as nn

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded