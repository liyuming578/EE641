from torch import nn

from Transformer.embedding.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        pos_emb = self.pos_emb(x)
        return self.drop_out(x + pos_emb)
