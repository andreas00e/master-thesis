import os 
import glob
import yaml 
import h5py
import tqdm 
import random 
import numpy as np 
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import norm 

import seaborn as sns 
import multiprocessing as mp
import time  

import itertools 
from functools import wraps

from typing import Tuple
from dataclasses import dataclass

from utils import time_it

@dataclass 
class PlotConfig(): 
    figsize: Tuple[int, int] = (10,5)

@dataclass 
class HistogramConfig(): 
    density: bool = True 
    stacked: bool = True 
    edgecolor: str = 'black' 
    alplha: float = 0.75 

@dataclass 
class PiechartConfig(): 
    pass

plotConfig = PlotConfig()
histogramConfig = HistogramConfig()
piechartConfig = PiechartConfig()

def plot(figsize: Tuple[int, int], label: str, xlabel: str, ylabel: str, figname: str):
    def plot_wrapper(func): 
        @wraps(func)
        def wrapper(*args, **kwargs): 
            fig = plt.figure(figsize=figsize)

            func(*args, **kwargs)

            plt.grid(color='grey', linestyle='dashed')
            plt.legend()
            plt.title(label=label)
            plt.xlabel(xlabel=xlabel)
            plt.ylabel(ylabel=ylabel)
            plt.tight_layout()
            plt.savefig(f'{figname}.png')

            return None 
        return wrapper  
    return plot_wrapper


class Visualize(): 
    def __init__(self, path: Path, robot: str, n_bins: int, hist_args: dict):
        self.path = path 
        self.robot = robot
        self.n_bins = n_bins

        self.files = [x for x in glob.glob(path+'*.hdf5') if robot in x]

        self.x, self.y, self.z, self.theta, self.gripper = [], [], [], [], []
        self.vars = {'x': self.x, 'y': self.y, 'z':self.z, 'theta': self.theta}

        self.create()
        global anotherfunnyvariable
        anotherfunnyvariable = (15, 10)

        self.hist_args = hist_args = hist_args

    def open_hdf5(self, file): 
        with h5py.File(file, 'r') as hf: 
            actions = hf['policy']['actions']
            x = actions[:, 0]
            y = actions[:, 1]
            z = actions[:, 2]
            theta = actions[:, 3]

        return x, y, z, theta
    
    @time_it
    def create(self): 
        with mp.Pool(processes=mp.cpu_count()) as pool: 
            result = pool.map(func=self.open_hdf5, iterable=self.files)
        
        # assign values from opened hdf5 file to respective lists 
        self.vars['x'] = list(itertools.chain(*list(list(result[i][0]) for i in range(len(result)))))
        self.vars['y'] = list(itertools.chain(*list(list(result[i][1]) for i in range(len(result)))))
        self.vars['z'] = list(itertools.chain(*list(list(result[i][2]) for i in range(len(result)))))
        self.vars['theta'] = list(itertools.chain(*list(list(result[i][3]) for i in range(len(result)))))

    # @plot(figsize=plotConfig.figsize, label='hallo', xlabel='xlabel', ylabel='hello', figname='test_dataclass')
    @time_it
    def side_stacked_hist(self): 
        for i, (_, v) in enumerate(list(self.vars.items())[:-1]): 
            _, bins, _ = plt.hist(v, **self.hist_args, facecolor='blue')
            y = norm.pdf(x=bins, loc=np.mean(v), scale=np.std(v))
            plt.plot(bins, y, color='red', linestyle='--')
        
        return self.hist_args
        

def stacked_hist(path: Path, robot: str, percentage: float, n_bins: int):
    # TODO: 2025-03-12: Think about adding functionality to include gripper 
    # Should be binary -> Does not seem to be necessary to create histogram
    x, y, z, theta = [], [], [], []
    # x = y = z = theta = gripper = []
    vars = {'x': x, 'y': y, 'z':z, 'theta': theta}
    # vars = {'x': x, 'y': y, 'z':z, 'theta': theta, 'gripper': gripper}
    files = [x for x in glob.glob(path+'*.hdf5') if robot in x]
    _len = len(files)

    for i in tqdm.tqdm(range(round(percentage*_len))):
        if percentage != 1: 
            i = random.randint(0, _len-1)
        
        file = files[i]
        with h5py.File(file, 'r') as hf: 
            actions = np.array(hf['policy']['actions'])
            x.extend(list(actions[:, 0]))
            y.extend(list(actions[:, 1]))
            z.extend(list(actions[:, 2]))
            theta.extend(list(actions[:, 3]))
            # if actions.shape[1] == 5: 
            #      gripper.extend(list(actions[:, 3]))
    
    # values = list(vars.values())
    # keys = list(vars.keys())

    # _, bins, _ = plt.hist(x=values[:3], bins=n_bins, density=True, histtype='bar', align='mid',
    #             stacked=True, color=plt.cm.Blues(np.linspace(0.2, 0.8, 3)), edgecolor='black', alpha=0.75, label=keys[:3])
    
    # for i in range(3): 
    #     y = norm.pdf(x=bins, loc=np.mean(values[i]), scale=np.std(values[i]))
    #     plt.plot(bins, y, color='red')

    # plt.legend()
    # plt.title('Stacked historgram using Matplotlib')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{robot}_{percentage*100}_{n_bins}_stacked_hist.png')


def hist(path: Path, robot: str, percentage: float, num_bins: int):
    # TODO: 2025-03-12: Think about adding functionality to include gripper 
    # Should be binary -> Does not seem to be necessary to create histogram
    x, y, z, theta = [], [], [], []
    # x = y = z = theta = gripper = []
    vars = {'x': x, 'y': y, 'z':z, 'theta': theta}
    # vars = {'x': x, 'y': y, 'z':z, 'theta': theta, 'gripper': gripper}
    files = [x for x in glob.glob(path+'*.hdf5') if robot in x]
    _len = len(files)

    for i in tqdm.tqdm(range(round(percentage*_len))):
        if percentage != 1: 
            i = random.randint(0, _len-1)
        
        file = files[i]
        with h5py.File(file, 'r') as hf: 
            actions = np.array(hf['policy']['actions'])
            x.extend(list(actions[:, 0]))
            y.extend(list(actions[:, 1]))
            z.extend(list(actions[:, 2]))
            theta.extend(list(actions[:, 3]))
            # if actions.shape[1] == 5: 
            #      gripper.extend(list(actions[:, 3]))

    n_rows = n_cols = 2 
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(14, 10))  

    for i, (k, v) in enumerate(vars.items()): 
        _, bins, _ = ax[int(i/2), i%2].hist(v, density=True, stacked=True, bins=num_bins, facecolor='blue', edgecolor='black', alpha=0.75)
        y = norm.pdf(size=bins, loc=np.mean(v), scale=np.std(v))
        ax[int(i/2), i%2].plot(bins, y, color='red', linestyle='--')
        ax[int(i/2), i%2].set_axisbelow(True)
        ax[int(i/2), i%2].grid(color='grey', linestyle='dashed')
        ax[int(i/2), i%2].set_title(f"Distribution of action values for {k}")
        ax[int(i/2), i%2].set_xlabel('Value')
        ax[int(i/2), i%2].set_ylabel('Frequency')
    
    fig.tight_layout()
    fig.savefig(f'{robot}_{percentage*100}_{num_bins}_hist.png')

def main():
    try: 
        with open('../configs/data.yaml', 'r') as file: 
            data = yaml.safe_load(file)

    except FileNotFoundError: 
        print("Requested yaml-file could not be found!")

    load_path = os.path.expanduser(data['load_path'])

    robot = 'kuka'
    global plotconfig
    plotconfig = PlotConfig(figsize=(10, 5))

    hist_args = data['plot']['histogram']['matplotlib']

    # stacked_hist(path=load_path, robot=robot, percentage=1, n_bins=25)

    visualize = Visualize(path=load_path, robot='kuka', n_bins=25, hist_args=hist_args)
    visualize.side_stacked_hist()

    # seaborn_hist(path=load_path, robot=robot, percentadge=1, n_bins=25)


    # files = [x for x in glob.glob(load_path+'*.hdf5')]

    # robots_count = dict([(robot, 0) for robot in robots])
     
    # for file in tqdm.tqdm(files): 
    #     with h5py.File(file, 'r') as hf: 
    #         robots_count[hf['metadata'].attrs['robot']] += 1 
    
    # robots_count = {'sawyer': 68112, 'widowx': 5050, 'baxter': 18054, 'kuka': 1608, 'franka': 7873, 'R3': 56720, 'fetch': 5000}
    # colors = ['red', 'blue', 'orange', 'green', 'yellow', 'grey', 'grey']

    # plt.pie(list(robots_count.values()), labels=list(robots_count.keys()), autopct='%1.1f%%', colors=colors)
    # plt.savefig('pie_chart.png')
    # plt.show()
            
if __name__ == '__main__': 
    main() 