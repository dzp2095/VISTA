"""Model package initialization and registration."""

from ..registry import register_model
from .unet import UNet

__all__ = [
    "UNet"
]