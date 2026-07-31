import os 
import h5py 


def main(): 
    file_dir = "~/ehrensberger/master-thesis/imports/mimicgen/datasets/robot"
    file_dir = os.path.expanduser(file_dir)
    
    files = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if "depth" not in file]
    
    for file in files: 
        with h5py.File(file, "r") as hf: 
            data = hf["data"]
            
            for demo in data.keys(): 
                joint_pos = data[demo]["obs"]["robot0_joint_pos"][()]
                joint_vel = data[demo]["obs"]["robot0_joint_vel"][()]
                print(joint_pos.shape)
                print(joint_vel.shape)
                
                
                
                
                exit()

if __name__ == "__main__": 
    main() 