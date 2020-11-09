import numpy as np
import torch
from typing import Union


def build_contact_map_from_pairs(mapping: list, max_length: int = 600) -> torch.Tensor:
    """
    Builds the contact map matrix A from a list of pairs of contacts

    Args:
        - mapping (list): list of pairs of indices (a, b)
        - max_length (int): maximum sequence length in the dataset
    """
    A = np.zeros((max_length, max_length))
    for pair in mapping:
        a, b = pair
        A[a, b] = 1
        A[b, a] = 1
    return A


def onehot_to_index(seq: Union[np.ndarray, list]) -> np.ndarray:
    """
    transforms a onehot encoding sequence for the RNA strand to an index
        sequence, where:
            0: A
            1: U
            2: C
            3: G
            4: no nucleotide
    Args:
        - seq: sequence of one hot encodings
    """
    return np.array(
        [
            np.where(onehot)[0][0] if len(np.where(onehot)[0]) > 0 else 4
            for onehot in seq
        ]
    )
