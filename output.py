import torch
import numpy as np
from torch import nn
from Transformer.model.transformer import Transformer
import pandas as pd

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

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    input_numpy = (np.load('features_full.npy').astype(np.float32))[100]
    input_tensor = torch.from_numpy(input_numpy).unsqueeze(0)

    input_tensor = input_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        predictions = (probs >= 0.5).float()
        print("Predicted output:", predictions)

    predictions_np = predictions.cpu().numpy()
    output_df = pd.DataFrame(predictions_np.reshape(predictions_np.shape[1], predictions_np.shape[2]))
    output_df.to_csv('model_output.csv', index=False)
    print("Output saved to model_output.csv")