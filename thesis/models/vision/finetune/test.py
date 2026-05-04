import torch.nn as nn 
from r3m import load_r3m

r3m = load_r3m("resnet18") # resnet18, resnet34
r3m.eval()

# Freeze every but the last layer
# for name, module in r3m.named_modules(): 
#     if 'layer4' not in name: 
#         if isinstance(module, nn.Conv2d): 
#             for param in module.parameters(): 
#                 param.requires_grad = False 
#         elif isinstance(module, nn.BatchNorm2d): 
#             module.eval() 
#             for param in module.parameters(): 
#                 param.requires_grad = False 
    
#     elif 'layer4' in name: 
#         if isinstance(module, nn.Conv2d): 
#             for param in module.parameters(): 
#                 param.requires_grad = True
#         elif isinstance(module, nn.BatchNorm2d):
#             module.train() 
#             module.reset_running_stats()
#             module.reset_parameters()
#             for param in module.parameters(): 
#                 param.requires_grad = True 

for name, module in r3m.named_modules(): 
    print(name, module)
    print('-----------------------')
    