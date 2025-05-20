import cv2
import datetime
import numpy as np
from tqdm import tqdm


import robosuite as suite
from robosuite.controllers import load_controller_config

import torch 
from torchvision.transforms import v2
from torchvision.transforms.functional import rotate

from models.autoencoder.autoencoder import DownsampleCVAE
from models.diffusion.diffuser import DownsampleObsLDM


controller_config = load_controller_config(default_controller="OSC_POSE")

# camera='agentview'
camera = 'agentview'
time=str(datetime.datetime.now())
env_name='Stack'

transform = v2.Compose([
    v2.ToImage(),                      # Convert to (C, H, W), dtype uint8
    v2.Resize((64, 64)),               # Resize
    v2.ToDtype(torch.float32, scale=False),  # Convert to float32 and divide by 255
    v2.Normalize(mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]),  # Normalize
])

def main(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create environment instance
    env = suite.make(
        env_name=env_name,
        robots='Panda',
        controller_configs=controller_config, 
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # camera_names=["agentview", "birdview", "robot0_eye_in_hand", "sideview"],
        camera_names=['agentview', 'robot0_eye_in_hand', 'robot0_robotview'],
        camera_heights=[1080, 64],
        camera_widths=[1080, 64],
        camera_depths=False,
    )
    
    print(env.sim.model.camera_names)

    # Load trained model 
    checkpoint_path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/thesis/logs/pretrain_diffusion/best_epoch/checkpoints/best_diffusion.ckpt'
    model = DownsampleObsLDM.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()

    # Reset the environment
    obs = env.reset()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{env_name}_{camera}_{time}.mp4", fourcc, 20.0, (1080, 1080))

    # agentview_image = transform(obs['agentview_image'])
    # rotate(agentview_image, angle=180)

    # robot0_eye_in_hand_image = transform(obs['robot0_eye_in_hand_image'])
    # rotate(robot0_eye_in_hand_image, angle=180)

    horizon = 4
    
    task_instruction = "Set the blue cube above the green cube."
    
    for i in tqdm(range(250), colour='green'):

        with torch.no_grad(): 
            agentview_image = transform(obs['agentview_image'])
            robot0_eye_in_hand_image = transform(obs['robot0_eye_in_hand_image'])
            rotate(robot0_eye_in_hand_image, angle=180)
            # agentview_image = agentview_image[None, ...].to(device)
            # robot0_eye_in_hand_image = robot0_eye_in_hand_image[None, ...].to(device)
            images = torch.cat((agentview_image[None, ...], robot0_eye_in_hand_image[None, ...]), dim=1)

            action = model.predict_action(task_instruction, images.to(device))
            action = action[0, :horizon, :].cpu().numpy()
            for i in range(horizon): 
                a = action[i, :]
                obs = env.step(a)[0]  # take action in the environment
                image = obs["{}_image".format(camera)]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.rotate(image, cv2.ROTATE_180)
                writer.write(image)

    env.close()
    writer.release()
    print(f"Saved {env_name}_{camera}_{time}.mp4")

if __name__ == '__main__': 
    main() 