import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
   
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
        label: TensorType["n, k"], 
        idxs: TensorType["n"],
        task: TensorType["n"], 
        robot: TensorType["n"], 
        fname: str
        ) -> None:
                                
        x = x.numpy() # [n, k]
        label_max = label.max(-1).values.numpy() # [n]
        label_idx = label.argmax(-1).numpy() # [n]
        idxs = idxs.numpy() # [n]
        task = task.numpy() # [n]
        robot = robot.numpy() # [n]
        
        unique_labels = np.unique(label_idx)
        palette = sns.color_palette("rocket", n_colors=len(unique_labels))
        color_map = dict(zip(np.unique(label_idx), palette))
        rgba_colors = np.array([mcolors.to_rgba(color_map[l]) for l in label_idx]) # assign a base RGB color to each category
        rgba_colors[:, 3] = np.clip(label_max, 0.05, 1.0) # assign opacity to each sample based on assignment confidence 
            
        x_2d = self.tsne.fit_transform(x) # [n, 2]
        
        df = pd.DataFrame({
            "x": x_2d[:, 0], 
            "y": x_2d[:, 1], 
            "idxs": idxs,  
            "label": label_idx,
            "task": pd.Series(task.astype(int)).map(TASK_DICT).fillna(pd.Series(task.astype(int)).astype(str)),
            "robot": pd.Series(robot.astype(int)).map(ROBOT_DICT).fillna(pd.Series(robot.astype(int)).astype(str))
            })
         
        fig, ax = plt.subplots(figsize=(10, 6))        
        ax = sns.scatterplot(
            data=df, 
            x="x",
            y="y", 
            hue=df.index, 
            palette=list(rgba_colors),
            size="idxs", 
            style="robot", 
            sizes=(10, 100), 
            edgecolor="none", 
            ax=ax
            )
        
        handles, texts = ax.get_legend_handles_labels()
        n_samples = len(df)
        kept_handles = handles[n_samples + 1 :]
        kept_texts = texts[n_samples + 1 :]

        label_handles = [
            Line2D([], [], marker="o", linestyle="None", color=color_map[l], label=str(l))
            for l in unique_labels
        ]
        label_title = [Line2D([], [], linestyle="None", label="label")]

        ax.legend(
            handles=label_title + label_handles + kept_handles,
            labels=["label"] + [str(l) for l in unique_labels] + kept_texts,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            ncol=2,
            frameon=True,
        )

        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)