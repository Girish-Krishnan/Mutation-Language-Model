"""Mutation Language Model package."""

from .model import BigramLanguageModel
from .mapping import encode, decode, vocab_size
from .dataset import load_dataset

__all__ = [
    "BigramLanguageModel",
    "encode",
    "decode",
    "vocab_size",
    "load_dataset",
]
