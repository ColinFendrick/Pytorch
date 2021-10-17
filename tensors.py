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
    return tensor

rand = generate_tensor(torch.rand)
ones = generate_tensor(torch.ones)
zeros = generate_tensor(torch.zeros)

def get_tensor_attributes(tensor):
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    return tensor

get_tensor_attributes(rand)
