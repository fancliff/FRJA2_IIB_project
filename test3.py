import torch

data = [1,2,3,4,5]

tensor = torch.tensor(data)

print(tensor)

print(f"Tensor shape: {tensor.shape}")
print(f"Tensor data type: {tensor.dtype}")

x = torch.rand(5,3)
print(x)