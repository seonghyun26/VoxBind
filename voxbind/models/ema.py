import torch
import torch.nn as nn
from copy import deepcopy


class ModelEma(nn.Module):
    """
    Exponential Moving Average (EMA) wrapper for a PyTorch model.
    inspired by https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
    Args:
        model (nn.Module): The model to be wrapped.
        decay (float, optional): The decay rate for the moving average. Defaults to 0.9999.
        device (torch.device, optional): The device on which to perform EMA. Defaults to None.

    Attributes:
        module (nn.Module): A copy of the model for accumulating the moving average of weights.
        decay (float): The decay rate for the moving average.
        device (torch.device): The device on which EMA is performed.

    Methods:
        update(model): Updates the EMA weights using the provided model.
        set(model): Sets the EMA weights to be the same as the provided model.
    """

    def __init__(self, model, decay=0.9999, device=None, foreach=False):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        # foreach=True fuses the per-step lerp into two multi-tensor kernels
        # (torch._foreach_mul_/_foreach_add_) instead of a Python loop that
        # allocates a fresh tensor per parameter. Numerically equivalent for
        # floating buffers; non-float buffers (e.g. int counters) are hard-copied.
        self.foreach = foreach
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def _update_foreach(self, model, decay):
        with torch.no_grad():
            ema_float, model_float = [], []
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if ema_v.is_floating_point():
                    ema_float.append(ema_v)
                    model_float.append(model_v)
                else:                       # int/bool buffers: EMA is undefined → hard copy
                    ema_v.copy_(model_v)
            if ema_float:
                # ema = decay*ema + (1-decay)*model, fused & in-place on the EMA storages
                torch._foreach_mul_(ema_float, decay)
                torch._foreach_add_(ema_float, model_float, alpha=1.0 - decay)

    def update(self, model):
        """
        Updates the EMA weights using the provided model.

        Args:
            model (nn.Module): The model to update the EMA weights with.
        """
        if self.foreach:
            self._update_foreach(model, self.decay)
        else:
            self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        """
        Sets the EMA weights to be the same as the provided model.

        Args:
            model (nn.Module): The model to set the EMA weights to.
        """
        self._update(model, update_fn=lambda e, m: m)
