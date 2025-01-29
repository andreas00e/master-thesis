import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import slideflow as sf 
import tfrecord 


def main(): 
    if "cmu_play_fusion" in tfds.list_builders(): 
        print("Yes")
    else: 
        print("No")

def data(): 
    ds = tfds.load('cmu_play_fusion', split='all', shuffle_files=False)
    assert isinstance(ds, tf.data.Dataset)
    print(ds)
    
def build(): 
    ds = tfds.load('cmu_play_fusion', split='all')
    ds = ds.take(1)  # Only take a single example

    for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
        print(list(example.keys()))
        image = example["image"]
        label = example["label"]
        print(image.shape, label)

def inspect_dataset():
    data = tfrecord.tfrecord_loader('cmu_play_fusion-train.tfrecord-00000-of-00064')

    print(type(data))


if __name__ == "__main__":
    inspect_dataset()
