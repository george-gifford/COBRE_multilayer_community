#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:50:16 2019

@author: George Gifford
email: george.w.gifford@kcl.ac.uk
"""

# Load packages
import numpy as np
import pandas as pd
import pickle

# Create a random matrix of proportions similar to a temporal co-occurrence matrix
rand_matrix = np.zeros([264,264])
rand_matrix[np.diag_indices(264)] = 1
rand_matrix[np.triu_indices(264, k=1)] = np.random.randint(1,10, size= len(np.triu_indices(264, k=1)[0]))/10
rand_matrix[np.tril_indices(264)] = rand_matrix.T[np.tril_indices(264)]

# Import node resting state network assignments 
# These are available from https://www.jonathanpower.net/2011-neuron-bigbrain.html
Neuron2011 = pd.read_excel('Neuron_consensus_264.xlsx', header = 1)
rsn_assignments = Neuron2011['Unnamed: 36']
rsn_names = np.unique(rsn_assignments)

# Between resting state network synchronisation function
def between_RSN_sync(temporal_c_matrix, rsn_assignments, rsn_names):
    '''
    This returns a symmetric matrix of mean temporal co-occurrence values of nodes
    shared by pairs of resting state networks.
    temporal_c_matrix: symmetric temporal co-occurrence matrix
    '''
    
    rsn_temporal_c = np.zeros([len(rsn_names),len(rsn_names)])
    for i in range(len(rsn_names)):
        for j in range(len(rsn_names)):
            rsn_temporal_c[i,j] = np.mean(temporal_c_matrix[rsn_assignments == rsn_names[i],:][:,rsn_assignments == rsn_names[j]])

    # Return a dataframe with resting state network labels
    rsn_temporal_c_df = pd.DataFrame(rsn_temporal_c, columns = rsn_names, index = rsn_names)
    return(rsn_temporal_c_df)

# Out of resting state network synchronisation function
def out_of_RSN_sync(temporal_c_matrix, rsn_assignments, rsn_names):
    '''
    This gives mean temporal co-occurence values of the nodes a resting state
    network shares, exluding nodes from the same network.
    temporal_c_matrix: symmetric temporal co-occurrence matrix
    '''
    
    dict_rsn_means = {}
    for r in range(12):
        dict_rsn_means[rsn_names[r]] = np.mean(rand_matrix[rsn_assignments == rsn_names[r],:][:,rsn_assignments != rsn_names[r]])
    
    # Return a dataframe with resting state network labels
    return(pd.DataFrame.from_dict(dict_rsn_means, orient='index', columns = ['out of RSN sync']))
    
# Run the above funcitons
between_rsn_example = between_RSN_sync(rand_matrix, rsn_assignments, rsn_names)
out_of_rsn_example = out_of_RSN_sync(rand_matrix, rsn_assignments, rsn_names)

# Save output
with open('between_rsn_example.pickle', 'wb') as f:
    pickle.dump(between_rsn_example, f)

with open('out_of_rsn_example.pickle', 'wb') as f:
    pickle.dump(out_of_rsn_example, f)




    
    
    
