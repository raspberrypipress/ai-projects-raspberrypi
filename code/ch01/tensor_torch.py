import torch
import torchvision.transforms as T
from PIL import Image

# pip install torch torchvision pillow

# Create a 1D tensor
vector = torch.tensor([1, 2, 3])
print(vector.shape)

# Create a 2D tensor
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix.shape)

# Create a 3D tensor of with height, width, and random RGB values 
image = torch.rand(3, 64, 64)
transform = T.ToPILImage()
img = transform(image)
print(img)
img.show()