import os 
import h5py
import numpy as np 
import tqdm 

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from typing import List, Union
from pathlib import Path

from matplotlib import pyplot as plt 

import cv2 

transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True), # [0, 255] -> [0.0, 1.0]
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] → [-1,1]
])

class FinetuneR3MData(Dataset):
    def __init__(self, load_dir: Path, tasks: Union[str, List], views: List): 
        super().__init__()
        self.load_dir = os.path.expanduser(load_dir)
        self.tasks = tasks
        self.transforms = transform
        self.views = views 
        
        if isinstance(tasks, str): 
            self.files = [file for file in os.listdir(self.load_dir) if file == self.tasks]
            assert len(self.files)!=0, 'Given task not in list of available tasks!'
        elif isinstance(tasks, list): 
            self.files = [file for file in os.listdir(self.load_dir) if file in self.tasks]
        else: 
            raise TypeError("self.task of unhandled type!")
        
        self.demo_count = dict()
        self.image_map = []
        
        for file in self.files: 
            with h5py.File(os.path.join(self.load_dir, file), 'r') as hf: 
                data = hf['data']
                views = [key for key in data['demo_0']['next_obs'].keys() if 'image' in key and key in self.views]
                keys = data.keys()
                self.demo_count[file] = len(keys) # Should be 1000 demos per hdf5 file 
                for view in views: 
                    for demo in data.keys(): 
                        image = data[demo]['next_obs'][view][()]
                        self.image_map.extend((file, view, demo, index) for index in range(image.shape[0]))
        
        print(views)
       
    def __len__(self): 
        return len(self.image_map) # 431646

    def __getitem__(self, idx):
        input = self.image_map[idx]
        return self.load_image(*input)
    
    def load_image(self, file, view, demo, index):
        
        # Min and max depth value were obtained by iterating over the respective dataset 
        norm_values = {'robot0_eye_in_hand': [0.0219, 3.6922], 'agentview': [0.3135, 3.0535]} 
        with h5py.File(os.path.join(self.load_dir, file), 'r') as hf: 

            rgb_image = hf['data'][demo]['next_obs'][view][index, ...]
            rgb_image = self.transforms(torch.from_numpy(rgb_image).permute(2, 0, 1))
       
            depth_image = hf['data'][demo]['next_obs'][view.replace('image', 'depth')][index, ...]
            print(view)
            
            min_depth, max_depth = norm_values[view.replace('_image', '')]
            print(min_depth)
            print(max_depth)

            depth_image = (depth_image-min_depth)/(max_depth-min_depth)
            # depth_image = (depth_image-min_depth)/(max_depth-min_depth) * 255.0

            # depth_image = depth_image*2-1 
            depth_image = torch.from_numpy(depth_image).permute(2, 0, 1).float() # (84, 84, 1) -> (1, 84, 84)
            depth_image = depth_image.expand(3, -1, -1) # (1, 84, 84) -> (3, 84, 84)
            
                
            # return rgb_image, depth_image 
            return depth_image
    
    
def main(): 
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    tasks = ['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5']
    # views = ['robot0_eye_in_hand_image', 'agentview_image']   
    views = ['agentview_image'] 
    # agentview 
    # min = 0.3135
    # max = 3.0535
    # robot0_eye_in_hand 
    # min = 0.0219
    # max = 3.6922
    
    
     
    dataset = FinetuneR3MData(load_dir=load_dir, tasks=tasks, views=views)
    # rgb_image, depth_image = dataset[10000]
    depth_image = dataset[954]
    # rgb_image = np.transpose(rgb_image.detach().cpu().numpy(), axes=(1, 2, 0)) 
    # rgb_image = (rgb_image+1)/2
    # print(np.max(rgb_image))
    # print(np.min(rgb_image))
    print(torch.min(depth_image))
    print(torch.max(depth_image))
    depth_image = np.transpose(depth_image.detach().cpu().numpy().astype(np.uint8), axes=(1, 2, 0)) 
    plt.imsave(fname='agent_depth.png', arr=depth_image)
    
    print(depth_image.shape)
    
    depth_image = cv2.applyColorMap(src=depth_image, colormap=cv2.COLORMAP_JET)
    plt.imsave(fname='agent_jet_depth.png', arr=cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))

    # min = 1000000000000
    # max = -10000000000000
    
    # print(len(dataset))
    
    # for a in tqdm.tqdm(range(len(dataset))):
    #     i = dataset[a]
    #     if torch.min(i) < min: 
    #         min = torch.min(i)
    #     if torch.max(i) > max: 
    #         max = torch.max(i)
            
    # print(min)
    # print(max)
    # image = dataset[10093]
    
    # print(torch.min(image))
    # print(torch.max(image))
    # print(torch.min(image[:3, ...]))
    # print(torch.max(image[:3, ...]))
    
if __name__ == '__main__': 
    main()