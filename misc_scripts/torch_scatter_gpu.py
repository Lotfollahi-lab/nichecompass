from torch_scatter import scatter_max
import torch

src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).to("cuda")
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]).to("cuda")
out = src.new_zeros((2, 6))

out, argmax = scatter_max(src, index, out=out)

print(out)
print(argmax)