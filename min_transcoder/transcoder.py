"""

Most of this is from https://github.com/jacobdunefsky/transcoder_circuits/blob/master/transcoder_training/sparse_autoencoder.py which
is in turn mostly from Authur Conmy's https://github.com/ArthurConmy/sae/blob/main/sae/model.py

"""

import gzip
import os
import pickle
from functools import partial
import dataclasses

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import transformer_lens
import transformer_lens.hook_points

# from .geom_median.src.geom_median.torch import compute_geometric_median

from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

# TODO(bschoen): Move save / load related functions to utils
# TODO(bschoen): Without hooked transformer?


@dataclasses.dataclass
class TranscoderConfig:
    # Input dimension of the transcoder
    d_in: int

    # Output dimension of the transcoder
    d_out: int

    # Hidden dimension of the transcoder
    d_hidden: int

    # Data type for tensor operations
    dtype: torch.dtype

    # Device to run the model on (e.g., 'cpu' or 'cuda')
    device: torch.device


@dataclasses.dataclass
class TranscoderResults:
    transcoder_out: Float[Tensor, "... d_out"]
    hidden_activations: Float[Tensor, "... d_hidden"]


class Transcoder(transformer_lens.hook_points.HookedRootModule):
    """

    Note:
        For simplicity, we remove:
        - resample_neurons_anthropic
        - collect_anthropic_resampling_losses
        - resample_neurons_l2
        - initialize w/ geometric median

    """

    def __init__(
        self,
        cfg: TranscoderConfig,
    ) -> None:

        super().__init__()

        self.cfg = cfg
        self.d_in = cfg.d_in

        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_hidden = cfg.d_hidden
        self.dtype = cfg.dtype
        self.device = cfg.device

        # transcoder stuff
        self.d_out = self.d_in
        self.d_out = cfg.d_out

        # NOTE: if using resampling neurons method, you must ensure that we initialise
        #       the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc: Float[Tensor, "d_in d_hidden"] = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.d_in,
                    self.d_hidden,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        )
        self.b_enc: Float[Tensor, "d_hidden"] = nn.Parameter(
            torch.zeros(
                self.d_hidden,
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.W_dec: Float[Tensor, "d_hidden d_out"] = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.d_hidden,
                    self.d_out,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec: Float[Tensor, "d_in"] = nn.Parameter(
            torch.zeros(
                self.d_in,
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.b_dec_out: Float[Tensor, "d_out"] = nn.Parameter(
            torch.zeros(
                self.d_out,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # setup hookpoints so we can easily use with transformer lens
        self.hook_transcoder_in = transformer_lens.hook_points.HookPoint()
        self.hook_hidden_pre = transformer_lens.hook_points.HookPoint()
        self.hook_hidden_post = transformer_lens.hook_points.HookPoint()
        self.hook_transcoder_out = transformer_lens.hook_points.HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "... d_in"]) -> TranscoderResults:
        # move x to correct dtype
        x = x.to(self.dtype)

        # Remove encoder bias as per Anthropic
        transcoder_in: Float[Tensor, "... d_in"] = self.hook_transcoder_in(
            x - self.b_dec
        )

        hidden_pre: Float[Tensor, "... d_hidden"] = self.hook_hidden_pre(
            einops.einsum(
                transcoder_in,
                self.W_enc,
                "... d_in, d_in d_hidden -> ... d_hidden",
            )
            + self.b_enc
        )
        hidden_activations: Float[Tensor, "... d_hidden"] = self.hook_hidden_post(
            torch.nn.functional.relu(hidden_pre)
        )

        transcoder_out: Float[Tensor, "... d_out"] = self.hook_transcoder_out(
            einops.einsum(
                hidden_activations,
                self.W_dec,
                "... d_hidden, d_hidden d_out -> ... d_out",
            )
            + self.b_dec_out
        )

        return TranscoderResults(
            transcoder_out=transcoder_out,
            hidden_activations=hidden_activations,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        """
        Update grads so that they remove the parallel component
            (d_hidden, d_in) shape

        """

        parallel_component: Float[Tensor, "d_hidden"] = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_hidden d_out, d_hidden d_out -> d_hidden",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_hidden, d_hidden d_out -> d_hidden d_out",
        )
