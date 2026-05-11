"""
Trainers module
"""

from .ours_trainer import OursTrainer
from .seg_trainer import SegTrainer

__all__ = [
    'OursTrainer',
    'SegTrainer',
]
