import os 
import h5py
import multiprocessing as mp 

# write a function that combines the depth and color information of  one task 
def main(): 
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    load_dir = os.path.expanduser(load_dir)
    
    task = 'stack'
    
    depth_files, color_files = [], []
    for f in os.listdir(load_dir):
        if task in f and f.endswith('.hdf5') and 'three' not in f: # TODO: Change for later tasks. Use simple implementation for now
            if 'depth' in f:
                depth_files.append(f)
            else:
                color_files.append(f)    
        
    depth_files = sorted(depth_files, key=lambda x: int(x.split('_')[1].replace('d', '')))
    color_files = sorted(color_files, key=lambda x: int(x.split('_')[1].split('.')[0].replace('d', '')))
    
    # with h5py.File(os.path.join(load_dir, color_files[-1]), 'r') as hf: 
    #     for key in hf.keys(): 
    #         print(key)
    
    for depth_file, color_file in zip(depth_files[1:], color_files[1:]): 
        with h5py.File(os.path.join(load_dir, depth_file), 'r') as dhf: 
            depth_data = dhf['data'].keys()
            with h5py.File(os.path.join(load_dir, color_file), 'r') as chf: 
                color_data = chf['data'].keys()
    
if __name__ == '__main__': 
    main()