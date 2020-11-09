import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class DSN(nn.Module):
    def __init__(self, seq_length: int = 600, depth: int = 256):
        super().__init__()
        self.seq_encoder = SequenceEncoder(depth)
        self.output_layer_symm = OutputLayersSymm(depth*6, (768, 512, 256))

    def forward(self, batch, masks):
        return self.output_layer_symm(self.seq_encoder(batch, masks))


class OutputLayersSymm(nn.Module):
    """
    OutputLayers with symetrization operation
        forward returns U(x) -> notation from the E2Efold paper
    """

    def __init__(
            self,
            in_channels: int = 1536,
            hidden_dims: Tuple[int, ...] = (768, 512, 256),
    ):
        super(OutputLayersSymm, self).__init__()

        convs = []
        for i, dim in enumerate(hidden_dims):
            convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=dim,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                )
            )
            in_channels = dim
        convs.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                stride=1,
                padding=1,
                kernel_size=3,
            )
        )

        self.convs = nn.Sequential(*convs)

    def forward(self, batch):
        """
        Parameters
        ----------
        batch (L, N, 3 * D)

        Returns
        -------
        scores U matrix of dim (N, L, L)
        """
        # (L, N, 3 * D) -> (N, L, 3 * D)
        x = batch.permute(1, 0, 2)

        n, l, d = x.size()

        x_img = torch.zeros((n, l, l, 2 * d))
        # TODO: make this more efficient
        for i in range(l):
            for j in range(l):
                x_img[:, i, j, :] = torch.cat((x[:, i, :], x[:, j, :]), dim=1)
        x_img = x_img.permute(0, 3, 1, 2)
        U_asym = self.convs(x_img)
        U_sym = U_asym + torch.transpose(U_asym, 2, 3)
        return U_sym.squeeze(1)


class SequenceEncoder(nn.Module):
    def __init__(
            self,
            depth: int = 256,
            sparse: bool = False,
            n_heads: int = 8,
            n_encoder_layers: int = 3,
            dim_feedforward=600,
    ):
        super(SequenceEncoder, self).__init__()
        self.nucleotide_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=depth,
            sparse=sparse,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * depth,  # nucleotide embeddings + pos embeddings
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )

        self.pos = PositionalEncoder(depth)

    def forward(self, batch, masks):
        """
        Parameters
        ----------
        batch (L, N): The batch of sequences to analyze each nucleotide is given
            a number (0, 1, 2, 3, or 4) as an index for the embeddings, where:
                0: A
                1: U
                2: C
                3: G
                4: no nucleotide
        masks (L, N): since sequences are different sizes masks hide the end
            of sequences from the model

        Returns
        -------
        encoded sequence of size (L, N, 3 * D)

        """
        x = self.nucleotide_embedding(batch)
        x = x.permute(1, 0, 2)
        x = self.pos(x).float()
        x = self.encoder(x, src_key_padding_mask=masks)
        x = self.pos(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, depth: int = 256, concatenate: bool = True):
        super(PositionalEncoder, self).__init__()
        self.concatenate = concatenate
        if depth % 2 != 0:
            raise ValueError(f"depth {depth} can't be odd")
        self.depth = depth

    def forward(self, batch):
        """
        Parameters
        ----------
        batch (L, N, D): batch containing sequences

        Returns
        -------
        batch with positional encoding with dimensions
            (L, N, 2 * D) if concatenating, and (L, N, D) if not concatenating

        L: sequence length
        N: batch size
        D: embedding dimension
        """
        l, n, d = batch.size()
        assert d % 2 == 0

        if self.concatenate:
            d = self.depth

        word_indices = np.arange(l)

        pes = []
        for t in word_indices:
            pe = []
            for k in range(d // 2):
                w = 1 / (10.0 ** (2 * k) / d)
                pe.append(np.sin(w * t))
                pe.append(np.cos(w * t))
            pes.append(pe)

        # (L, D)
        pes = torch.tensor(pes)

        # (L, D) -> (L, N, D)
        pes = pes.repeat(n, 1, 1).permute(1, 0, 2)

        if self.concatenate:
            batch = torch.cat((batch, pes), dim=2)
        else:
            batch += pes

        return batch
