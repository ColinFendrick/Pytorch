import torch
import numpy as np

data = [[1,2], [3,4]]
np_array = np.array(data)
np_tensor = torch.from_numpy(np_array)

print(f"{np_tensor} \n")

# Shaping tensors
my_shape = (2,3,)


def generate_tensor(generator_fn=torch.rand, shape=my_shape):
    tensor = generator_fn(shape)

    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

    print(f"Tensor from {generator_fn.__name__}: {tensor}")
    return tensor

def get_tensor_attributes(tensor=generate_tensor()):
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device} \n")
    return tensor

get_tensor_attributes(generate_tensor(torch.rand))
get_tensor_attributes(generate_tensor(torch.ones))
get_tensor_attributes(generate_tensor(torch.zeros))
