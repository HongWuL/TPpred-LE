import torch
import torch.nn as nn
from models.transfomer import *

class TransformerLEM(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    """

    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int, device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 ):
        super(TransformerLEM, self).__init__()

        self.d_model = d_model
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.device = device
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, d_model)

        self.input_embedding = nn.Embedding(20, d_model)
        self.label_embedding = nn.Embedding(15, d_model)

        self.position_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=2048,
                                     dropout=dropout) for _ in range(n_enc_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=2048,
                                     dropout=dropout) for _ in range(n_dec_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, key_mask, labels, att_mask=None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
        """
        if x.size(-1) == 1:
            x = x.squeeze(-1).long()
            x = self.input_embedding(x)
            x = self.position_encoding(x)
        else:
            x = self.lin(x)  # batch_size, len, d_model
            x = self.position_encoding(x)

        atts_x = []
        for i, encoder in enumerate(self.encoder_layers):
            x, att_x = encoder(x, src_key_padding_mask=key_mask)
            atts_x.append(att_x)

        atts_tgt = []
        atts_cross = []
        y = self.label_embedding(labels)
        for i, decoder in enumerate(self.decoder_layers):
            y, att_tgt, att_cross = decoder(y, x, tgt_mask=att_mask)
            atts_tgt.append(att_tgt)
            atts_cross.append(att_cross)


        return y, atts_x, atts_tgt, atts_cross


class TPMLC_single(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    """
    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int,  device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 ):
        super(TPMLC_single, self).__init__()

        self.rp = TransformerLEM(in_dim, out_dim, max_len, d_model, device, nhead, n_enc_layers, n_dec_layers, dropout)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, key_mask, labels, att_mask = None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
        """

        y, atts_x, atts_tgt, atts_cross = self.rp(x, key_mask, labels, att_mask)
        
        outputs = self.fc(y).squeeze(-1)

        return outputs, atts_x, atts_tgt, atts_cross

class TPMLC(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    """
    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int,  device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 ):
        super(TPMLC, self).__init__()

        self.rp = TransformerLEM(in_dim, out_dim, max_len, d_model, device, nhead, n_enc_layers, n_dec_layers, dropout)

        # self.fc = nn.Sequential(
        #         nn.Linear(d_model, 1)
        #         nn.Sigmoid()
        #     )

        self.fcs = nn.ModuleList([
            nn.Sequential(

                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            for _ in range(out_dim)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, key_mask, labels, att_mask = None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
        """

        y, atts_x, atts_tgt, atts_cross = self.rp(x, key_mask, labels, att_mask)
        # y:  (batch_size, n_class, d_model)

        outputs = []
        for i, fc in enumerate(self.fcs):
            output = fc(y[:, i, :])    # (batch_size, d_model) * (d_mode, 1)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=-1)

        # outputs = self.fc(y).squeeze(-1)

        
        return outputs, atts_x, atts_tgt, atts_cross

