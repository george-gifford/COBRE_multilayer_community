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

# Create a random array of integers between 1 and 5 (node x time windows)
community_array = np.random.randint(1,5, size= (264,10))

# Find flexibility function
def get_flexibility(communities):
    '''
    Returns a vector (length of the number of nodes) of the number of transitions 
    from one commmunity to another divided by N-1 time windows
    
    communities: An array of multilayer community assignments (node x number of time windows)
    '''
    n_nodes = len(communities)
    n_tws = len(communities[0,:])
    
    count = np.zeros([n_nodes])
    for n in range(n_nodes):
    	for t in range(n_tws-1):
    		if communities[n][t] != communities[n][t+1]:
    			count[n] = count[n] + 1
    		else:
    			pass
    	count[n] = count[n] / (n_tws - 1)
    return(count)

# Run get_flexibility()
flexibility = get_flexibility(community_array)

# Save file
with open('flexibility_file.pickle', 'wb') as f:
    pickle.dump(flexibility, f)




