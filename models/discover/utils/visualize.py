import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
   
import torch 
from torchtyping import TensorType


TASK_DICT = {
    0: "square",
    1: "threading"
}

ROBOT_DICT = {
    0: "iiwa", 
    1: "panda",
    2: "sawyer", 
    3: "ur5e"
}

map_task = np.vectorize(lambda x: TASK_DICT.get(x, str(x)))
map_robot =np.vectorize(lambda x: ROBOT_DICT.get(x, str(x)))


class Visualize(): 
    def __init__(
        self, 
        tsne_kwargs: DictConfig
        ) -> None: 
        
        self.tsne_kwargs = tsne_kwargs
        self.tsne = TSNE(**self.tsne_kwargs)
        
    def plot_(
        self, 
        x: TensorType["n", "k"],
        label: TensorType["n"], 
        idxs: TensorType["n"],
        task: TensorType["n"], 
        robot: TensorType["n"], 
        fname: str
        ) -> None:
                                
        x = x.numpy() # [n, k]
        label = label.argmax(-1).numpy() # [n]
        idxs = idxs.numpy() # [n]
        task = task.numpy() # [n]
        robot = robot.numpy() # [n]
            
        x = self.tsne.fit_transform(x) # [n, 2]
        
        df = pd.DataFrame({
            "x": x[:, 0], 
            "y": x[:, 1], 
            "idxs": idxs,  
            "label": label, 
            })
        
        df["task"] = map_task(task.astype(int))
        df["robot"] =  map_robot(robot.astype(int))
         
        plt.figure(figsize=(8, 6))
        scatterplot = sns.scatterplot(
            data=df, 
            x="x",
            y="y", 
            hue="robot", 
            style="task", 
            size="idxs", 
            sizes=(10, 100)
            )
        
        fig = scatterplot.get_figure() 
        fig.savefig(fname)
        plt.close()