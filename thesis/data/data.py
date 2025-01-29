import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds

def main(): 
    ds = tfds.load("jaco_play", split="all")
    ds = ds.take(1)
    print(ds)
    print(type(ds)) 

if __name__ == "__main__":
    main() 
