#!/usr/bin/env python

import numpy as np
import torch

import matplotlib.pyplot as plt

# Code inspired from https://github.com/tensorflow/playground/blob/master/src/dataset.ts

def classify_gaussian(nsamples=100):
    '''
        Generate Gaussian distributed points with positive samples around (2, 2)
        and negative samples around (-2, -2).
        
        Inputs:
            nsamples: Number of samples to generate. Should be even
            
        Outputs:
            data: (nsamples, 2) dimensional data
            labels: (nsamples,) dimensional label (0 or 1)
    '''
    labels = np.zeros(nsamples)
    labels[nsamples//2:] = 1
    
    data = np.random.randn(nsamples, 2)
    
    data[:nsamples//2, 0] += 2
    data[:nsamples//2, 1] += 2
    
    data[nsamples//2:, 0] -= 2
    data[nsamples//2:, 1] -= 2
    
    return data, labels

def classify_circle(nsamples=100):
    '''
        Generate data distributed as two concentric circles.
        
        Inputs:
            nsamples: Number of samples to generate. Should be even
            
        Outputs:
            data: (nsamples, 2) dimensional data
            labels: (nsamples,) dimensional label (0 or 1)
    '''
    labels = np.zeros(nsamples)
    labels[nsamples//2:] = 1
    
    # Data inside the first circle
    rad = np.random.rand(nsamples//2, 1)*0.5
    angles = np.random.rand(nsamples//2, 1)*2*np.pi

    x1 = rad*np.sin(angles)
    y1 = rad*np.cos(angles)
    
    data1 = np.hstack((x1, y1))
    
    # Data inside the second circle
    rad = 0.7 + np.random.rand(nsamples//2, 1)*0.3
    angles = np.random.rand(nsamples//2, 1)*2*np.pi

    x2 = rad*np.cos(angles)
    y2 = rad*np.sin(angles)
    
    data2 = np.hstack((x2, y2))
    
    data = np.vstack((data1, data2))
    
    labels = np.hypot(data[:, 0], data[:, 1]) < 0.5
    
    return data, labels

def classify_spiral(nsamples=100):
    '''
        Generate data distributed on spirals.
        
        Inputs:
            nsamples: Number of samples to generate. Should be even
            
        Outputs:
            data: (nsamples, 2) dimensional data
            labels: (nsamples,) dimensional label (0 or 1)
    '''
    labels = np.zeros(nsamples)
    labels[nsamples//2:] = 1
    
    data = np.zeros((nsamples, 2))
    
    n2 = nsamples / 2
    for idx in range(nsamples//2):
        r = idx / n2 * 5
        t = 1.75 * idx / n2 * 2 * np.pi 
        data[idx, 0] = r*np.sin(t)
        data[idx, 1] = r*np.cos(t)
        
    for idx in range(nsamples//2):
        r = idx / n2 * 5
        t = 1.75 * idx / n2 * 2 * np.pi + np.pi
        data[idx + nsamples//2, 0] = r*np.sin(t)
        data[idx + nsamples//2, 1] = r*np.cos(t)
     
    return data, labels

def get_data(dataset_name, nsamples=100, noise=0.0):
    '''
        Generate data for a given dataset name
        
        Inputs:
            dataset_name: One of 'gauss', 'circle', 'spiral'
            nsamples: Number of samples to generate
            noise: Amount of noise to add to data
            
        Outputs:
            data: (nsamples, 2) dimensional data
            labels: (nsamples,) dimensional labels
    '''
    if dataset_name == 'gauss':
        data, labels = classify_gaussian(nsamples)
    elif dataset_name == 'circle':
        data, labels = classify_circle(nsamples)
    elif dataset_name == 'spiral':
        data, labels = classify_spiral(nsamples)
    else:
        raise ValueError('%s dataset not implemented'%dataset_name)
    
    data += np.random.randn(nsamples, 2)*noise
    
    return data.astype(np.float32), labels.astype(np.float32)   

if __name__ == '__main__':
    data, labels = get_data('spiral', 1000, 0.1)
    
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()