#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:31:06 2017

@author: George Gifford
email: george.w.gifford@kcl.ac.uk

"""

# Load packages
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns; sns.set()
import pickle
import matplotlib.pyplot as plt

# A list of participants
ps = ['p1','p2','p3','p4','p5']

# Create a dictionary of participants each with a random array of integers 
# between 1 and 5 (node x time windows)
communities_ps = {}
for p in ps:
    communities_ps[p] = np.random.randint(1,6, size= (264,10))

# Create vectors of community node assignments
def create_community_vectors(sample_of_community_assignments):
    '''
    Takes a dictionary of community assignment arrays (node x time windows)
    and returns a n_samples x n_features array, where each sample is a community
    from a participant and each feature is a proportion of times a node was 
    assigned that community.
    
    This function also returns an indexer for mapping vectors back to participant
    community assignments.
    
    sample_of_community_assignments: original commmunity assignments
    '''
    
    # Get list of participants
    ps = list(sample_of_community_assignments.keys())
    
    # Create vectors (N nodes) for each community of the proportion of times that
    # community is assigned to each node.
    vectors_ps = {}
    for p in ps:
    	community_array = sample_of_community_assignments[p]
    	communities = {}
    	for community in range(np.ptp(community_array)+1):
    		vector_communities = np.zeros([len(community_array)])
    		for node in range(len(community_array)):
    			vector_communities[node] = (community_array[node,:] == community+1).sum()/len(community_array[node,:])
    		communities[community+1] = vector_communities
    	vectors_ps[p] = communities
    
    
    # Concatenate into one array (n_samples, n_features): all_vectors
    # Create an indexing system in order to map clustering results back to the 
    # orginal community assignments: index_for_vector
    
    all_vectors = np.zeros([1,264])
    
    index_for_vector = {}
    x = 0
    for p in ps:
    	community_index = {}
    	for community in range(len(vectors_ps[p])):
    		all_vectors = np.concatenate((all_vectors,np.reshape(vectors_ps[p][community+1],(1,len(community_array)))),axis=0)
    		x = x +1
    		community_index[community+1] = x
    	index_for_vector[p] = community_index
    
    return(all_vectors[1:,],index_for_vector)

vectors, v_index = create_community_vectors(communities_ps)

# Find the approriate K using inertia scores
def kmeans_find_k(inputfile):
    '''
    Computes inertia scores from K means clustering, using K values from 2 to 20
    
    inputfile: n_samples x n_features array
    '''
    inertia_values = {}
    for n_clusters in range(2,21):
        kmeans = KMeans(n_clusters=n_clusters,n_init=500, random_state=150,n_jobs =-10)
        kmeans.fit_predict(inputfile)
        inertia_values[n_clusters] = kmeans.inertia_
        print(n_clusters)
    inertia_values = pd.DataFrame.from_dict(inertia_values, orient="index")
    inertia_values['K'] = list(range(2,21))
    inertia_values.columns = ['Inertia','K']
    inertia_values.to_csv('inertia_values.csv')
    f = plt.figure(figsize=(10,6))
    sns.set_style('whitegrid')
    sns.set(font="Liberation Serif",style="whitegrid")
    sns.set_context("paper", rc={"font.size":25,"axes.titlesize":25,"axes.labelsize":25,
                                 'xtick.labelsize': 20, 'ytick.labelsize': 25,'legend.fontsize': 25,
                                 'lines.linewidth': 2}) 
    lp = sns.lineplot(x='K',y='Inertia',data=inertia_values,color="black",ax=f.add_subplot(111))
    lp.set_xticks(np.arange(2,21,2))
    sns.despine(f)
    f.savefig('inertia_values.png',dpi=300)
    print('Find k done.')

kmeans_find_k(vectors)


# Run the K means clustering algorithm
def kmeans_func(inputfile,v_index,k,sample_of_community_assignments):
    '''
    Function that runs the k means algorithm.
    
    inputfile: n_samples x n_features array
    v_index: indexer to map from a n_samples x n_features to orignal community assignments
    k: clustering level
    sample_of_community_assignments: original community assignments
    '''
    
    # Run the k means algorithm
    kmeans = KMeans(n_clusters=k,n_init=500, random_state=123,n_jobs =-5)
    kmeans.fit_predict(inputfile)
    k_centres = kmeans.cluster_centers_
    k_labels = kmeans.labels_
    
    # Get list of participants
    ps = list(sample_of_community_assignments.keys())
    
    # Get the orignial community assignments and re-populate with the k means 
    # clustered community assignments
    participants_clustered_communities = {}
    for p in ps:
        p_communities = sample_of_community_assignments[p]
        empty_mat = np.zeros([len(p_communities),len(p_communities[0,:])])
        for community in range(np.amax(p_communities)):
            c = community+1
            i,j = np.where(p_communities == c)
            new_c = k_labels[v_index[p][community+1]-1]+1
            empty_mat[i,j] = new_c
        participants_clustered_communities[p] = empty_mat
        
    k_centres_df = pd.DataFrame(k_centres,index = [list(range(1,k+1))])
    return(participants_clustered_communities, k_centres_df)

clustered_communities_ps, cluster_centres  = kmeans_func(vectors, v_index, 3, communities_ps)

# Save cluster assignment file
with open('k_means_clustered_community_assignments.pickle', 'wb') as f:
    pickle.dump(clustered_communities_ps, f)

# Save csv file of cluster centres
cluster_centres.to_csv('k_means_cluster_centres.csv',header=False, )


