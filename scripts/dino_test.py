import os 
import torch

from models.discover.skill_encoder import SkillEncoder

print(f"Working directory: {os.getcwd()}")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

rand = torch.rand(size=(10, 3, 84, 84))

out = model(rand)

print(out.shape)
