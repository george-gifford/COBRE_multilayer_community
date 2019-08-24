#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:50:16 2019

@author: George Gifford
email: george.w.gifford@kcl.ac.uk
"""

# Load packages
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Create a random array of integers between 1 and 5
community_array = np.random.randint(1,5, size= (264,10))

# Temporal co-occurrence function
def get_temporal_c(communities):
    '''
    Returns a node x node matrix of temporal co-occurrence values, which is the 
    proportion of times two nodes share the same community across time windows.
    
    communities: An array of multilayer community assignments (node x number of time windows)
    '''
    
    n_nodes = len(communities)
    n_tws = len(communities[0,:])
    mat = np.zeros([n_nodes,n_nodes])
    
    for t in range(n_tws):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if communities[i,t] == communities[j,t]:
                    mat[i,j]= mat[i,j] + 1
                else:
                    pass
    mat = mat / n_tws
    return(mat)

# Run get_temporal_c()
temporal_c = get_temporal_c(community_array)

# Plot a heatmap of the results
sns.heatmap(temporal_c)
plt.show()

# Save file
with open('temporal_c_file.pickle', 'wb') as f:
    pickle.dump(temporal_c, f)