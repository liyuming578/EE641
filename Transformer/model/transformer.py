import torch
from torch import nn
from Transformer.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, in_channels, out_channels, in_frames, out_frames, d_model, n_head, max_len,
                 ffn_hidden, n_layers, num_class, drop_prob, device):
        super().__init__()
        self.device = device
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm([in_frames, out_channels])
        self.relu1 = nn.ReLU()
        self.conv1d_frame = nn.Conv1d(in_frames, out_frames, kernel_size=1)
        self.layer_norm2 = nn.LayerNorm([out_frames, out_channels])
        self.relu2 = nn.ReLU()

        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.num_class = num_class
        # MLP for final output
        self.mlp = nn.Sequential(
            nn.Linear(d_model, out_channels // 2),  # d_model is the output dimension of the Transformer
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, num_class)  # Assume 10 is the number of output classes
        )

    def forward(self, x):
        x = self.conv1d(x).transpose(1, 2)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        # x = x.transpose(1, 2)
        x = self.conv1d_frame(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)

        identity_mask = self.make_mask(x)
        output = self.encoder(x, identity_mask)
        batch_size, seq_len, dim = output.shape

        output = self.mlp(output.reshape(-1, dim))
        output = output.reshape(batch_size, seq_len, self.num_class)

        return output

    def make_mask(self, x):
        image_mask = torch.ones(x.size(0), 1, 1, x.size(1), device=x.device)
        return image_mask


if __name__ == "__main__":
    in_channels = 256
    out_channels = 64
    in_frames = 600
    out_frames = 60
    d_model = 64
    n_head = 16
    max_len = 60
    ffn_hidden = 2048
    n_layers = 4
    num_class = 10
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Transformer
    model = Transformer(in_channels=in_channels, out_channels=out_channels, in_frames=in_frames, out_frames=out_frames,
                        d_model=d_model,
                        n_head=n_head,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_layers=n_layers,
                        num_class=num_class,
                        drop_prob=drop_prob,
                        device=device).to(device)

    batch_size = 4
    seq_len = 600
    random_input = torch.randn(batch_size, 256, seq_len).to(device)

    output = model(random_input)
    print("Model output shape:", output.shape)
    print("Transformer model ran successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
