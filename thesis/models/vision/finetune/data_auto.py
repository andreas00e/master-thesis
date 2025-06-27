import os 
import h5py
import numpy as np 
import tqdm 
import pickle as pkl 

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
    def __init__(self, load_dir: Path, tasks: Union[str, List], views: List, use_transform: bool=True): 
        super().__init__()
        self.load_dir = os.path.expanduser(load_dir)
        self.tasks = tasks
        self.transforms = transform
        self.views = views 
        self.use_transform = use_transform
        
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
        
        # print(views)
       
    def __len__(self): 
        return len(self.image_map) # 431646

    def __getitem__(self, idx):
        input = self.image_map[idx]
        return self.load_image(*input)
    
    def load_image(self, file, view, demo, index):
        # print(file, view, demo)
        
        # Min and max depth value were obtained by iterating over the respective dataset 
        # norm_values = {'robot0_eye_in_hand': [0.0219, 3.6922], 'agentview': [0.03135, 3.0535]} # absolute values 
        norm_values = {'robot0_eye_in_hand': [0.04186, 1.04219765], 'agentview': [0.35653934, 3.0535419]} # smoothed values 

        with h5py.File(os.path.join(self.load_dir, file), 'r') as hf: 

            rgb_image = hf['data'][demo]['next_obs'][view][index, ...]
            if self.use_transform: 
                rgb_image = self.transforms(torch.from_numpy(rgb_image).permute(2, 0, 1))
            else: 
                rgb_image = torch.from_numpy(rgb_image).float().permute(2, 0, 1)
            
            # Convert depth map to rgb image
            depth_image = hf['data'][demo]['next_obs'][view.replace('image', 'depth')][index, ...]         
            min_depth, max_depth = norm_values[view.replace('_image', '')]
            # min_depth = np.min(depth_image)
            max_depth = np.max(depth_image)
            # print(f'min_depth: {np.min(depth_image)}')
            # print(f'max_depth: {np.max(depth_image)}')
            # depth_image = (depth_image-min_depth)/(max_depth-min_depth) # [ ,  ] -> [0, 1]
            depth_image = torch.from_numpy(depth_image).permute(2, 0, 1).float() # (84, 84, 1) -> (1, 84, 84)
            # depth_image = torch.clamp(input=depth_image, min=0.0, max=1.0)
            depth_image = depth_image.expand(3, -1, -1) # (1, 84, 84) -> (3, 84, 84)
            depth_image = depth_image # [0, 1] -> [0, 255.0]
            
            return rgb_image, depth_image
    
    
def main(): 
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    # tasks = ['stack_d1_depth.hdf5']
    tasks = ['stack_d0_depth.hdf5', 'stack_d1_depth.hdf5']
    # views = ['robot0_eye_in_hand_image', 'agentview_image']   
    # views = ['robot0_eye_in_hand_image', 'agentview_image']   
    views = ['robot0_eye_in_hand_image'] 
    # agentview 
    # min = 0.3135
    # max = 3.0535
    # robot0_eye_in_hand 
    # min = 0.0219
    # max = 3.6922
    
    dataset = FinetuneR3MData(load_dir=load_dir, tasks=tasks, views=views)
    # print(len(dataset))
    
    min_file = f'thesis/models/vision/finetune/{views[0]}_min.pkl'
    max_file = f'thesis/models/vision/finetune/{views[0]}_max.pkl'
    if os.path.isfile(min_file): 
        with open(min_file, 'rb') as minf:
            min_depth = pkl.load(minf) 
    if os.path.isfile(max_file): 
        with open(max_file, 'rb') as maxf:
            max_depth = pkl.load(maxf) 
    else: 
        min_depth, max_depth = [], [] 
        for _ , depth_image in tqdm.tqdm(dataset, colour='green'): 
            min_depth.append(float(torch.min(depth_image)))
            max_depth.append(float(torch.max(depth_image)))
    
        min_depth = np.array(min_depth)
        max_depth = np.array(max_depth)
        
        with open(min_file, 'wb') as minf: 
            pkl.dump(min_depth, minf)
        with open(max_file, 'wb') as maxf: 
            pkl.dump(max_depth, maxf)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))
    counts0, bins0, patches0 = axs[0].hist(x=min_depth, bins=30, color='skyblue', edgecolor='black')
    axs[0].set_title(f'Minimal depth distribution')
    print(f'Min counts: {counts0}')
    print(f'Min bins: {bins0}')
    
    counts1, bins1, patches1 = axs[1].hist(x=max_depth, bins=30, color='skyblue', edgecolor='black')
    axs[1].set_title(f'Maximal depth distribution')
    print(f'Max counts: {counts1}')
    print(f'Max bins: {bins1}')
    fig.suptitle(f'Min/Max distributions for {views[0]}')
    plt.tight_layout()
    fig.savefig(f'thesis/models/vision/finetune/{views[0]}_dist.png', dpi=300)
    
    print(counts1[10])
    print(bins1[11])
    exit()

    # rgb_image, depth_image = dataset[10000]
    rgb_image, depth_image = dataset[7032]

    # rgb_image = np.transpose(rgb_image.detach().cpu().numpy(), axes=(1, 2, 0)) # (84, 84, 3) -> (3, 84, 84)
    # rgb_image = (rgb_image+1)/2 # [-1, 1] -> [0, 255]
    
    # plt.imsave(fname='robot0_image.png', arr=rgb_image)
    # exit()
    

    depth_image = np.transpose(depth_image.detach().cpu().numpy(), axes=(1, 2, 0)) 
    
    # print(f"min torch depth: {np.min(depth_image)}")
    #print(f"max torch depth: {np.max(depth_image)}")
    
    # print(f"rgb shape: {rgb_image.shape}")
    # print(f"depth shape: {depth_image.shape}")
    plt.imsave(fname='robot_depth.png', arr=depth_image)
    
    print(depth_image.shape)
    depth_image = depth_image * 255.0
    depth_image = cv2.applyColorMap(src=depth_image[:, :, 0].astype(np.uint8), colormap=cv2.COLORMAP_JET)
    plt.imsave(fname='robot_depth_color.png', arr=cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))
    
    
if __name__ == '__main__': 
    main()