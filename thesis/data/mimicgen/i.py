import os 
import glob 
import h5py
import cv2 
import tqdm
import numpy as np

import sys
sys.path.append('/home/ubuntu/ehrensberger/CLIP')


from clip.clip import load, tokenize
# from RoLD.RoLD import utls
import torch 



instructions = [
    "Could you please place the blue cube on the green one?",
    "Would you mind stacking the blue cube on top of the green cube?",
    "Please put the blue cube over the green cube.",
    "I’d like the blue cube to go on top of the green one.",
    "Can you make sure the blue cube sits on the green cube?",
    "Pick up the blue cube and gently place it on the green cube.",
    "Lift the blue cube and set it on top of the green cube.",
    "Find the green cube and stack the blue cube right on top.",
    "First, locate the green cube. Then place the blue cube on it.",
    "Carefully place the blue cube so it’s resting above the green cube.",
    "The goal is to have the blue cube sitting on top of the green cube.",
    "Arrange the cubes so the blue one ends up on the green one.",
    "Make a small stack with the green cube on the bottom and the blue cube on top.",
    "Position the cubes so that the blue is above the green.",
    "Finish the task by putting the blue cube on the green cube.",
    "Place the blue cube directly over the green cube.",
    "Stack the blue cube on top of the green one.",
    "Put the blue cube right on top of the green cube.",
    "Set the blue cube above the green cube.",
    "Position the blue cube on the green cube.",
    "Let’s stack the blue cube on top of the green cube.",
    "Try placing the blue cube on the green one.",
    "How about we put the blue cube above the green one?",
    "You can stack the blue cube on top of the green cube now.",
    "I think the blue cube should go over the green one.",
    "Put the blue cube right above the green one.",
    "Set the blue cube down on top of the green cube.",
    "Please arrange the blue cube so it’s on the green cube.",
    "Place the blue cube neatly on the green cube.",
    "Carefully stack the blue cube above the green one.",
    "Move the blue cube to rest on the green cube.",
    "Stack the blue block over the green block.",
    "Gently position the blue cube on the green cube.",
    "The blue cube should go on top of the green one.",
    "Let the blue cube rest on the green cube.",
    "Place the blue cube on top — the green one should be underneath.",
    "Could you set the blue cube on the green cube for me?",
    "Make sure the blue cube is placed above the green one.",
    "The blue cube needs to sit on top of the green cube.",
    "Go ahead and put the blue cube right on top of the green cube.",
    "I need you to put the blue cube over the green one.",
    "Please stack the blue one directly on the green cube.",
    "Lay the blue cube gently on the green cube.",
    "The blue cube belongs on top of the green cube.",
    "Try stacking the blue cube right over the green cube.",
    "Gently balance the blue cube on the green one.",
    "Could you arrange the cubes so that the blue one sits on the green one?",
    "Let’s place the blue cube over the green block.",
    "Put the blue cube down so it’s sitting atop the green cube.",
    "Please ensure the blue cube is placed squarely on the green cube."
]

def clip(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = load('ViT-B/16', device=device)
    model.eval()

    print(len(instructions))

    text = tokenize(instructions).to(device)

    with torch.no_grad(): 
        text_features = model.encode_text(text)

        print(text_features.shape)


def main(): 
    load_dir = '~/ehrensberger/mimicgen/datasets/core'
    load_dir = os.path.expanduser(load_dir)

    task = 'pick_place'
    files = glob.glob(f"{os.path.join(load_dir, task)}*")
    
    for file in files:
        with h5py.File(file, 'r') as hf:
            data = hf['data']
            for demo in data.keys(): 
                keys = data[demo].keys()
                for subtask_term_signals_key in data[demo]['datagen_info']['subtask_term_signals'].keys():
                
                    subtask_term_signals = data[demo]['datagen_info']['subtask_term_signals'][subtask_term_signals_key][()]
                    # condition = subtask_term_signals == 1
                    # segment = np.argwhere(condition)
                    segment = np.where(subtask_term_signals==1)
                    print(subtask_term_signals_key)
                    print(segment)
                    # print(f'{subtask_term_signals_key}: {segment}')
                    # print(f'subtask_term_signal for task {subtask_term_signals_key}: {segment}')
                
                break
    # for file in files: 
    #     with h5py.File(file, 'r') as hf:
    #         data = hf['data']           
    #         for i, demo in enumerate(data.keys()):
    #             if i == 100: 
    #                 exit()
    #             # Define the codec and create VideoWriter object
    #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    #             writer = cv2.VideoWriter(f"thesis/scenes/scene_{i}.mp4", fourcc, 20.0, (84, 84)) 
               
    #             demo = data[demo]
    #             obs = demo['obs']['agentview_image'][()]
    #             for i in range(obs.shape[0]):
    #                 writer.write(obs[i, ...])
                
    #             writer.release()
    
    
if __name__ == '__main__': 
    main()