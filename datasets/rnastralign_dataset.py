import torch
from torch.utils.data import Dataset
from torch import Tensor
import pickle
import numpy as np

from typing import Tuple
from utils import build_contact_map_from_pairs, onehot_to_index
import collections


# Define the keys for the data object
X = 0
Y = 1
SEQ_LENGTH = 2
SOURCE = 3
CONTACT_PAIRS = 4
RNA_SS_data = collections.namedtuple(
    'RNA_SS_data',
    'seq ss_label length name pairs',
)


class RNAStralignDataset(Dataset):
    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
        # Actual sequences:
        data_x = np.array([instance[X] for instance in data])
        # pool = multiprocessing.Pool()
        onehot_to_index(data_x[0])
        self.data_x = list(map(onehot_to_index, data_x))
        # sequence lengths:
        self.seq_lengths = [instance[SEQ_LENGTH] for instance in data]
        # source of the sequence:
        self.source = [instance[SOURCE] for instance in data]
        # contact map pairs for each sequence:
        contact_pairs = [instance[CONTACT_PAIRS] for instance in data]
        self.contact_pairs = list(
            map(build_contact_map_from_pairs, contact_pairs)
        )

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx: int) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, str, int
    ]:
        """
        Returns
        -------
        Tuple of:
            - sequence in the form of [x_0, ..., x_L], where x_i in {0, 1, 2, 3, 4}
                where:
                    0: A
                    1: U
                    2: C
                    3: G
                    4: no nucleotide
            - tensor of the contact map solution
            - tensor with a key mask for the transformer encoder to not consider
                non-nucleotide entries
            - tensor of the M(x) matrix for this specific sequence
            - string for the source of the sequence
            - size of the sequence < 600
        """
        x = torch.tensor(self.data_x[idx])
        mask = torch.tensor(x == 4)
        a_star = torch.tensor(self.contact_pairs[idx])

        a_s = np.where(x == 0)[0]
        u_s = np.where(x == 1)[0]
        c_s = np.where(x == 2)[0]
        g_s = np.where(x == 3)[0]
        M_x = torch.zeros(600, 600)

        M_x[a_s[0], u_s[1]] = 1
        M_x[u_s[0], a_s[1]] = 1
        M_x[c_s[0], g_s[1]] = 1
        M_x[g_s[0], c_s[1]] = 1
        for i in range(600):
            for j in range(4):
                if i + j < 600:
                    M_x[i, i + j] = 0
                if i - j > 0:
                    M_x[i, i - j] = 0
        print((~mask).sum())
        print("oi")
        print(len(x))


        return (
            x,
            a_star,
            mask,
            torch.tensor(M_x),
            str(self.source[idx]),
            int(self.seq_lengths[idx]),
        )

