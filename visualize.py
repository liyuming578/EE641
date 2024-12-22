import numpy as np
import pandas as pd

file = np.load('labels_600_10.npy')[100]
df = pd.DataFrame(file)
df.to_csv("labels.csv", index=False)

