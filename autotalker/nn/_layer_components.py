import torch.nn as nn


class MaskedLinear(nn.Linear):
    """
    Masked linear component adapted from https://github.com/theislab/scarches. 
    """
    def __init__(self, n_input: int, n_output: int, mask, bias=True):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                             "n_input x n_output")
        super().__init__(n_input, n_output, bias)
        
        self.register_buffer("mask", mask.t())

        # Zero out weights with the mask so that the optimizer does not consider
        # them
        self.weight.data *= self.mask

    def forward(self, input):
        output = nn.functional.linear(input, self.weight * self.mask, self.bias)
        return output