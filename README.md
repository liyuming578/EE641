# Intelligent Urban Sound Detection

## Project Structure
Here is an overview of the project structure, outlining the main directories and files included in the Intelligent Urban Sound Detection project:

```plaintext
Intelligent Urban Sound Detection
│
├── dataset.py          # Handles data loading and preprocessing
├── evaluate.py         # Evaluation scripts to assess model performance
├── main.py             # Main executable script for the project
├── output.py           # Handles output generation
├── train.py            # Script for training models
├── visualize.py        # Visualization utilities for data
│
├── Dataset Creation and Preprocessing
│   ├── Data_Download.ipynb       # Script to download the dataset
│   ├── Dataset_Creation.ipynb    # Script to create the new audio sound events
│   ├── Wav_Datacreation.ipynb    # Script to create the original audio file dataset
│   └── Pre-processing.ipynb      # Script to preprocess the audio and create the MFCC dataset
│
├── Transformer
│   ├── model
│   │   ├── encoder.py      # Encoder component of the model
│   │   └── transformer.py  # Transformer model architecture
│   │
│   ├── embedding
│   │   ├── positional_encoding.py  # Positional encoding for transformer model
│   │   └── transformer_embedding.py
│   │
│   ├── layers
│   │   ├── layer_norm.py                # Layer normalization
│   │   ├── multi_head_attention.py      # Multi-head attention mechanism
│   │   ├── position_wise_feed_forward.py  # Position-wise feed-forward network
│   │   └── scale_dot_product_attention.py  # Scaled dot product attention
│   │
│   ├── blocks
│   │   └── encoder_layer.py  # Encoder layer of the transformer model

```

## Installation

To set up the Intelligent Urban Sound Detection project, install the required dependencies. Below is a list of essential packages with specified versions:

- **Anaconda**: Manage environments and install multiple Python packages. (`anaconda=2023.09`)
- **PyTorch**: For flexible and fast neural networks. (`pytorch=2.1.2`)
- **Scikit-Learn**: For machine learning tasks. (`scikit-learn=1.4.2`)
- **Numpy**: For numerical computing. (`numpy=1.26.3`)
- **Pandas**: For data manipulation and analysis. (`pandas=2.2.2`)
- **Librosa**: For audio signal processing. (`librosa=0.10.0`)
- **Mutagen**: For handling audio metadata. (`mutagen=1.47`)
- **IPython**: For interactive computation. (`ipython=8.16.0`)
- **Matplotlib**: For visualization and plotting. (`matplotlib=3.8.0`)
- **Plotly**: For interactive plots and dashboards. (`plotly=5.18.0`)
- **Soundfile**: For reading and writing sound files. (`soundfile=0.12.1`)
- **Pydub**: For manipulating audio files. (`pydub=0.25.1`)
- **Tqdm**: For progress bars. (`tqdm=4.66.0`)

```bash
conda install anaconda=2023.09 pytorch=2.1.2 scikit-learn=1.4.2 numpy=1.26.3 pandas=2.2.2 librosa=0.10.0 matplotlib=3.8.0 pillow=10.1.0 soundfile=0.12.1 tqdm=4.66.0 ipython=8.16.0 -c conda-forge
pip install mutagen plotly pydub
```
## Project Setup

To run the project, ensure that the path to the .npy format dataset is correctly specified and execute the main file.

## Configuration

The number of input frames (`in_frames`) and the number of features per frame (`in_features`) can be adjusted according to user-specific needs. Given that the label dimensions are 60x10, the output for each audio sample will also be internally adjusted by the neural network to match this size.

#### Data flow:

***MFCC Features:*** 
`5000 × 256 × 600` → `5000 × 64 × 600` → `5000 × 600 × 64` → `5000 × 60 × 64` → `5000 × 60 × 10`

***Original Features:*** 
`5000 × 1323000` → `5000 × 600 × 2205` → `5000 × 2205 × 600` → `5000 × 64 × 600` → `5000 × 600 × 64` → `5000 × 60 × 64` → `5000 × 60 × 10`

