import torch


class BinarizeFunction(torch.autograd.Function):
    """
    Straight-Through Estimator with {-1, +1} output.
    Forward: sign(x). Backward: gradient passes unchanged.
    Using {-1,+1} instead of {0,1} aligns cosine similarity with Hamming distance:
    cosine({-1,+1}^D) = 1 - 2*hamming_dist/D
    """

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x).float()  # {-1, +1}

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


binarize = BinarizeFunction.apply
