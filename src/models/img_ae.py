# TODO: which output activation function should be used?

from __future__ import annotations
from typing import Sequence, Tuple
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    channels: Sequence[int]      # e.g. [32, 64, 128]
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for c in self.channels:
            x = nn.Conv(features=c, kernel_size=(self.kernel_size,),
                        strides=(2,), padding="SAME", use_bias=True)(x)
            x = nn.relu(x)
        return x


class Decoder(nn.Module):
    channels: Sequence[int]      # e.g. [64, 32]
    out_channels: int            # e.g. 1
    kernel_size: int = 4
    act: callable = nn.relu
    out_act: callable | None = nn.tanh  # set to None or nn.sigmoid as needed

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Upsample through all hidden transpose-conv layers
        for c in self.channels:
            x = nn.ConvTranspose(features=c, kernel_size=(self.kernel_size,),
                                 strides=(2,), padding="SAME", use_bias=True)(x)
            x = self.act(x)
        # Final upsample to original channel count
        x = nn.ConvTranspose(features=self.out_channels, kernel_size=(self.kernel_size,),
                             strides=(2,), padding="SAME", use_bias=True)(x)
        if self.out_act is not None:
            x = self.out_act(x)
        return x


class Conv1DAutoencoder(nn.Module):
    enc_channels: Sequence[int]      # e.g. [32, 64, 128]
    in_channels: int = 1
    kernel_size: int = 4
    act: callable = nn.relu
    out_act: callable | None = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z = Encoder(self.enc_channels, kernel_size=self.kernel_size, act=self.act)(x)
        dec_channels = self.enc_channels[:-1][::-1]
        x_hat = Decoder(
            dec_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            act=self.act,
            out_act=self.out_act,
        )(z)
        return x_hat, z
