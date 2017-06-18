# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:31:58 2017


"""

'''
Module qui reutilise les fonctions de lecture de tensorflow pour reformatter les images dans MNIST
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np


def down_sample(image, nbits):
    down_sampled_image = np.zeros_like(image)
    for i in range(len(image)):
        #formule pour reduire le nombre de bits, elle marche
        down_sampled_image[i] = np.floor((image[i] * 255) / (2**(8-nbits)))/(2**nbits-1) 
    return down_sampled_image
    
def get_data(nbits, origin):
    down_sampled_images = np.zeros_like(origin.images)
    for i in range(len(down_sampled_images)):
        down_sampled_images[i] = down_sample(origin.images[i], nbits)
    return down_sampled_images, origin.labels
    
