import torch
import matplotlib.pyplot as plt
import numpy as np 

img = torch.rand(3, 84, 84)  # random RGB
img = img.permute(1, 2, 0).numpy()

print(np.max(img))
print(np.min(img))

plt.imsave(fname='rand.png', arr=img)