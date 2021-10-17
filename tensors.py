import torch
import numpy as np

data = [[1,2], [3,4]]
np_array = np.array(data)
np_tensor = torch.from_numpy(np_array)

print(np_tensor)

# Shaping tenors
my_shape = (2,3,)


def generate_tensor(generator_fn, shape=my_shape):
    tensor = generator_fn(shape)
    print(f"Tensor from {generator_fn.__name__}: {tensor}")

generate_tensor(torch.rand)
generate_tensor(torch.ones)
generate_tensor(torch.zeros)
