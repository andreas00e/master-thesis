import os 
import tqdm
from torch.utils.data import DataLoader, random_split
from data.mimic.data_mimicgen import MimicgenDataset
from matplotlib import pyplot as plt 

def main(): 
    args = dict()
    load_dir = '~/ehrensberger/mimicgen/datasets/core/'
    load_dir = os.path.expanduser(load_dir)
    robot = None # no robot specified for mimicgen core dataset
    task = ['stack_d1_depth.hdf5']
    # view = 'robot0_eye_in_hand_image' # Not neccessary as long as we include all (two) views
    action_horizon = 16
    image_horizon = 10

    args['load_dir'] = load_dir
    args['robot'] = robot
    args['task'] = task
    args['action_horizon'] = action_horizon
    args['image_horizon'] = image_horizon
    args['observations'] = 'forward'
    args['expand_depth'] = 'colormap'

    dataset = MimicgenDataset(**args)
    
    train_dataset, _, _ = random_split(dataset=dataset, lengths=[0.01, 0.01, 0.98])
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256)
    
    for train_data in train_dataloader: 
        image = train_data['image']
        agentview = image['agentview']
        robot0 = image['robot0_eye_in_hand']
        
        agentview = agentview[0, 0, 3:, ...].permute(1, 2, 0).numpy() / 255.0
        robot0 = robot0[0, 0, 3:, ...].permute(1, 2, 0).numpy() / 255.0

        plt.imsave(fname='robot0_eye_in_hand_d1.png', arr=robot0)
        plt.imsave(fname='agentview_d1.png', arr=agentview)

        
        break 
        
        
        
    exit()
        
    # images = next(iter(dataset))['image']
    
    # robot0_depth_image = images['robot0_eye_in_hand'][0, 3:, ...]
    # robot0_depth_image = robot0_depth_image.permute(1, 2, 0).detach().cpu().numpy()
    # agentview_depth_image = images['agentview'][0, 3:, ...]
    # agentview_depth_image = agentview_depth_image.permute(1, 2, 0).detach().cpu().numpy()
    # plt.imsave(fname='robot0_data.png', arr=robot0_depth_image)
    # plt.imsave(fname='agentview_depth_image.png', arr=agentview_depth_image)

    # max = 1 
    # min = -1 
    
    
    # for i in range(len(dataset)):
    #     action = dataset.__getitem__(i)['action']
    #     if torch.min(action) < min: 
    #         print(torch.min(action))
    #     if torch.max(action) > max: 
    #         print(torch.max(action))
    
    
    # print(dataset.traj_map[1000:1010])
    # exit()
    
    # for i, _ in tqdm.tqdm(enumerate(dataset)): 
    #    item = dataset.__getitem__(i)

    # image = next(iter(dataset))['image']
    # print(torch.max(image))
    # print(torch.min(image))
    # plt.imsave(fname='agentview.png', arr=torch.permute(image[:3, ...] / 255, dims=(1, 2, 0)).numpy())
    # plt.imsave(fname='robot0_eye_in_hand.png', arr=torch.permute(image[3:, ...] / 255, dims=(1, 2, 0)).numpy())
    
    
if __name__ == '__main__': 
    main() 
