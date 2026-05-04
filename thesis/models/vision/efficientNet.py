import os 
import h5py

from PIL import Image
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the weights
weights = EfficientNet_B0_Weights.IMAGENET1K_V1

# Get the associated transforms
preprocess = weights.transforms()

file = '~/ehrensberger/mimicgen/datasets/core/stack_d0_depth.hdf5'
file = os.path.expanduser(file)

with h5py.File(file, 'r') as hf: 
    img = hf['data']['demo_0']['next_obs']['agentview_image'][()][0, ...]

# Load an image
# img = Image.open("your_image.jpg").convert("RGB")

# Apply the preprocessing transforms
img = torch.from_numpy(img[None, ...]).permute(0, 3, 1, 2).to(device=device) 
img = preprocess(img)
print(f"Type of tensor: {type(img)}")
print(f"Shape of tensor: {img.shape}")

# img_tensor = preprocess(img).unsqueeze(0) # Add batch dimension
# img_tensor = img_tensor.to(device=device)

# Load the model with weights
model = efficientnet_b0(weights=weights).to(device=device)
model.eval()

# Run inference
with torch.no_grad():
    output = model(img)
