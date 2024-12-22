import torch
from torch.utils.data import DataLoader, random_split
from dataset import NPYDataset
from Transformer.model.transformer import Transformer
from train import train
import numpy as np

if __name__ == '__main__':
    in_channels = 256
    out_channels = 64
    in_frames = 600
    out_frames = 60
    d_model = 64
    n_head = 16
    max_len = 60
    ffn_hidden = 2048
    n_layers = 6
    num_class = 10
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_file = np.load('features_full.npy')
    label_file = np.load('labels_full.npy')

    dataset = NPYDataset(feature_file, label_file)

    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    train(model, train_dataloader, test_dataloader, device)

