import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO: construye aqui el modelo CNN a su consideracion.

    def forward(self, x):
        # TODO: implementa aqui el paso hacia adelante actualizando el input x del modelo. Por ejemplo:
        # x = self.conv1(x)
        # x = self.pool(x)
        # x = self.flatten(x)
        # logits = self.classifier(x), es el output del modelo antes de aplicar softmax
        return logits
