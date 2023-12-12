import torch
import torch.nn as nn
from typing import Iterable, List

# Code from CS330
class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = base_module

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### For randomly initializing the parameters, use torch.randn.
        ### Note: you should use nn.Parameter to wrap your parameters so that they are registered as
        ### learnable.
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###
        self.lora_A, self.lora_B = None, None
        ## YOUR CODE HERE, complete for Q2.2b
        in_size, out_size = base_module.weight.shape
        self.lora_A = torch.nn.Parameter(torch.zeros(in_size, lora_rank))
        self.lora_B = torch.nn.Parameter(torch.randn(out_size, lora_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###

        ## YOUR CODE HERE, complete for Q2.2b
        return base_out + (x @ self.lora_A) @ self.lora_B.T

def parameters_to_fine_tune(model: nn.Module, mode: str) -> Iterable[nn.Parameter]:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    For every mode except "all", the model is going to be GPT-2 (transformers.GPT2LMHeadModel).
    We encourage you to print the model architecture (e.g. by placing a PDB breakpoint and doing
    `print(model)`), and identifying where the model layers are located.

    Note: this function only returns the list of parameters to fine tune. It doesn't change the
    `requires_grad` component of any parameters, and you should not touch that in this assignment!

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)

    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    parameters_to_fine_tune: List[nn.Parameter] = None
    if mode == "all":
        # Every learnable parameter from `model` should be fine-tuned.
        # Complete this for Q0.1
        parameters_to_fine_tune = model.parameters()
    elif mode == "last":
        # Only fine tune the last 2 transformer blocks
        # Complete this for Q2.1
        parameters_to_fine_tune = [*model.transformer.h[-2].parameters(), *model.transformer.h[-1].parameters()]
    elif mode == "first":
        # Only fine tune the first 2 transformer blocks
        # Complete this for Q2.1
        parameters_to_fine_tune = [*model.transformer.h[0].parameters(), *model.transformer.h[1].parameters()]
    elif mode == "middle":
        # Only fine tune middle 2 transformer blocks
        # Complete this for Q2.1
        mid_ind = len(model.transformer.h)//2
        parameters_to_fine_tune = [*model.transformer.h[mid_ind - 1].parameters(), *model.transformer.h[mid_ind].parameters()]
    elif mode.startswith("lora"):
        # Only fine tune the rank decomposition matrices A and B from the LoRA layers.
        # Hint: consider using the `.modules()` function of nn.Module and checking for modules that
        # are an instance of LoRALayerWrapper.
        # Complete this for Q2.2c
        parameters_to_fine_tune = sum([[m.lora_A, m.lora_B] for m in model.modules() if isinstance(m, LoRALayerWrapper)],[])
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")

    return parameters_to_fine_tune