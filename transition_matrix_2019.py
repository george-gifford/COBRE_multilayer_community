#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:31:06 2017

@author: George Gifford
email: george.w.gifford@kcl.ac.uk
"""

# Load packages
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set number of communities
n_coms = 4

# Set number of nodes
n_nodes = 264

# Set number of time windows
n_tws = 10

# Create a random array of integers between 1 and 5 (node x time windows)
community_array = np.random.randint(1,n_coms+1, size= (n_nodes,n_tws))

# Make it so half of the community transitions are between 1 and 2
# (to make visualisation more interesting)
community_array[:,0] = 1
community_array[:,1] = 2
community_array[:,2] = 1
community_array[:,3] = 2
community_array[:,4] = 1
community_array[:,5] = 2

# Get a non-symetrical matrix of transitions (n communities x n communities)
trans_mat = np.zeros([n_nodes,n_coms,n_coms])
for node in range(n_nodes):
    for trans in range(len(community_array[0,:])-1):
        if community_array[node,trans] != community_array[node,trans+1]:
            trans_mat[node,int(community_array[node,trans])-1,
                      int(community_array[node,trans+1])-1] = trans_mat[node,
                         int(community_array[node,trans])-1,int(community_array[node,trans+1])-1] +1
        else:
            pass

# Make the transition matrices unidirectional by averging inward and outward transitions
trans_mat_undr = np.zeros([n_nodes,n_coms,n_coms])
for node in range(n_nodes):
    for i in range(n_coms):
        for j in range(n_coms):
            trans_mat_undr[node,i,j] = (trans_mat[node,i,j] + trans_mat[node,j,i]) / 2

# Collapse all nodes into one transition matrix (mean across nodes)
mean_trans_mat = np.mean(trans_mat,axis = 0)

# Plot the output
f = plt.figure(figsize=(7,7))
G = nx.from_numpy_array(mean_trans_mat,create_using = nx.Graph())
mappingG = {}
for x in range(n_nodes):
    mappingG[x] = x+1
G = nx.relabel_nodes(G,mappingG)
e_weights = list(d[2]['weight'] for d in list(G.edges(data=True)))
nx.draw(G, with_labels=True, node_color='lightgrey', node_size=1500, 
        width=np.asarray(e_weights)*2,pos=nx.circular_layout(G),
        ax=f.add_subplot(111), font_size=24)

# Save the figure
f.savefig('transition_matrix_example.png', dpi = 300)




