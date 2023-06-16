import torch
import torch.nn as nn

class CondDiscriminator(nn.Module):
    def __init__(self,
                 n_input,
                 n_conditions,
                 n_hidden,
                 leaky_relu_slope=0.2):
        """
        Conditional Discriminator class.
        """
        super(Discriminator, self).__init__()

        layers = []
        n_prev_layer = n_input

        for n in n_hidden:
            layers.append(nn.Linear(n_prev_layer, n))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            n_prev_layer = n

        layers.append(nn.Linear(n_prev_layer, n_conditions))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output