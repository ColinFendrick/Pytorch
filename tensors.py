import torch
import numpy as np

data = [[1,2], [3,4]]
np_array = np.array(data)
np_tensor = torch.from_numpy(np_array)

print(np_tensor)
