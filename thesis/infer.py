import os 
import torch 
import pickle as pkl 
from tqdm import tqdm
import lightning.pytorch as pl 
import numpy as np 

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt 
from sklearn.manifold import TSNE

from models.autoencoder.autoencoder import DownsampleCVAE
from data.robonet_dataset import RoboNetCustomizedDataset

from typing import List, Dict 
from torch.nn.utils.rnn import pad_sequence 
from itertools import chain 
import pandas as pd 

# TODO: Use multiprocessing/threading to speed up function execution 
def collate_fn(data: List[Dict]) -> Dict:
    batch = dict() 

    for key, value in data[0].items():
        sample = [d[key] if isinstance(value, str) else torch.tensor(d[key]) for d in data]
        if isinstance(value, str): 
            batch.update({key: sample.copy()})
        else: 
            batch.update({key: pad_sequence(sample.copy(), batch_first=True)}) 
        sample.clear()

    return batch

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'thesis/logs/lightning_logs/version_24/checkpoints/epoch=131-step=82236.ckpt'
    load_dir = '~/ehrensberger/RoboNetCustomized/hdf5/'
    load_dir = os.path.expanduser(load_dir)
    robots = ['kuka', 'widowx']

    if not os.path.exists('predictions.pkl') or os.path.getsize('predictions.pkl') == 0: 
        dataset = RoboNetCustomizedDataset(load_dir=load_dir, robots=robots)
        dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=20, collate_fn=collate_fn) # suggested max number of workers in current system: 20 
        print(os.getcwd())
        # state_dict = torch.load(checkpoint_path, weights_only=True)  # This is now safe
        model = DownsampleCVAE.load_from_checkpoint(checkpoint_path=checkpoint_path)
        trainer = pl.Trainer()
        # model.load_state_dict(state_dict=state_dict)
        predictions = trainer.predict(model=model, dataloaders=dataloader)

        predictions = list(chain.from_iterable(predictions))
        predictions = pd.DataFrame(predictions, columns=['robot', 'latent_vector'])

        with open('predictions.pkl','wb') as file: 
            pred = predictions.to_pickle(file)
            pkl.dump(pred, file)

    else: 
        with open('predictions.pkl', 'rb') as file: 
            predictions = pkl.load(file)
     
    # number of elements we want to take from the predictions dataframe for later visualization
    n = 100
    # take subset of predictions
    predictions = predictions[:n]
    # vertically stack individual arrays for data array 
    X = np.vstack(predictions['latent_vector'])
    labels = list(predictions['robot'])
    robot_color = {'widowx': 0.1, 'kuka': 0.5}
    group_mapping = {0.1: 'widowx', 0.5: 'kuka'}
    # list of colors for later scatter plot 
    colors = np.array([robot_color[l] for l in labels])

    tsne = TSNE(perplexity=30, random_state=42)
    latent = tsne.fit_transform(X=X[:n, :])

    fig, ax = plt.subplots()
    scatter = plt.scatter(x=latent[:, 0], y=latent[:, 1], s=5, c=colors, alpha=0.7)
    handles, labels = scatter.legend_elements(prop="colors", num=2)
    print(labels)
    exit()

    # Replace numerical color values with group names
    new_labels = list(set(predictions['robot']))
    # kw = dict(prop="colors", num=2)
    # ax = ax.legend(*scatter.legend_elements(**kw), title='Robots')
    ## plt.legend(handles, ['a', 'b'], title='Robots')
    plt.title("t-SNE Visualization of Latent Trajectory Vectors")
    plt.savefig('scatter.png')
if __name__ == '__main__': 
    main()