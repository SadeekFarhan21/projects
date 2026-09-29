"""smolgrad: reverse-mode autodiff and a small neural network library on numpy."""

from . import functional, nn, optim
from .gradcheck import gradcheck
from .random import manual_seed
from .tensor import Tensor, is_grad_enabled, no_grad, tensor

__all__ = ["Tensor", "tensor", "no_grad", "is_grad_enabled", "gradcheck", "manual_seed",
           "functional", "nn", "optim"]
__version__ = "0.1.0"
