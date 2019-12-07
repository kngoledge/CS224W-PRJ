
# coding: utf-8

# In[1]:


import snap
import pandas as pd
import numpy as np
from os import path
import networkx as nx
from collections import Counter
from __future__ import division


# In[2]:


df_with_clusters_classified = pd.read_csv('cmty_nodes.csv', sep='\t', index_col=0)
edges_df = pd.read_csv('edges.csv',sep='\t', index_col=0)
G = nx.from_pandas_edgelist(edges_df, 'u', 'v')
G = G.to_undirected()
nodes_G = set(edges_df['u'].values)
nodes_G = nodes_G.union(set(edges_df['v'].values))


# In[15]:


for x in xrange(0,289):
    num_rows = df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==x].shape[0]
    if num_rows > 1 and num_rows < 15: 
        print(x)


# In[16]:


df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==7]
# relatively same age, 750+, random blog videos with humor 
# About Section of community channel: Hi! I'm Natalie and I've been making videos online for far too long. They're (hopefully) fun videos that are a combination of monologue and sketches that focus on the humourous aspects of everyday life.  
# Ex. videos from kwa22122: "Apparently, I need to get laid...." and "STORYTIME: I GOT MY NAILS DID" and "File under: Gay Boy Problems"
# Ex. videos from Black McGrow: ("Simba Goes Home to Africa" & "There's a monkey on my face")


# In[18]:


df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==59]
# entertainment bloggers about anime games (pokemon, minecraft, YuGiOh) & produces original videos ("shorts" or "parodies" on their youtube channel); around the same age
# machinima About Me description: Machinima is the most notorious purveyor and cultivator of fandom and gamer culture. This channel features current popular content such as BFFs, Super Best F.
# TheNewLittleKuriboh has videos on YuGiOh anime parody
# soccertutor2 not found? 


# In[23]:


df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==271]
# music genre, international music, live music, relatively same age


# In[20]:


df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==231]
# asian bloggers & vloggers, weekly beauty tutorials & karate matches & double eyelid & asian culture 


# In[21]:


df_with_clusters_classified[df_with_clusters_classified['cnm_cmty']==261]
# wrestling matches e.g. "Jeff Hardy vs Umaga - RAW 19.11.07" and "Anderson Silva caught flying scissor heel hook". However, user may not necessarily be the original producer (hbknhhhfan has many wrestling videos as favorites, none originally uploaded)


# In[24]:


for x in xrange(0,289):
    num_rows = df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==x].shape[0]
    if num_rows > 1 and num_rows < 15: 
        print(x)


# In[26]:


df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==8]
# relatively same age, 750+, random blog videos with humor 
# About Section of community channel: Hi! I'm Natalie and I've been making videos online for far too long. They're (hopefully) fun videos that are a combination of monologue and sketches that focus on the humourous aspects of everyday life.  
# Ex. videos from kwa22122: "Apparently, I need to get laid...." and "STORYTIME: I GOT MY NAILS DID" and "File under: Gay Boy Problems"
# Ex. videos from Black McGrow: ("Simba Goes Home to Africa" & "There's a monkey on my face")


# In[28]:


df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==63]
# entertainment bloggers about anime games (pokemon, minecraft, YuGiOh) & produces original videos ("shorts" or "parodies" on their youtube channel); around the same age
# machinima About Me description: Machinima is the most notorious purveyor and cultivator of fandom and gamer culture. This channel features current popular content such as BFFs, Super Best F.
# TheNewLittleKuriboh has videos on YuGiOh anime parody
# soccertutor2 not found? 


# In[34]:


df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==248]
#WWE, WWF (e.g. "WLW World League Wrestling #102 Highlights")


# In[32]:


df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==247]
# gamers who record their PPV for wresting games (e.g. "SvR 2007 - WWE vs ROH: One Night Only" - https://www.youtube.com/watch?v=Lz98iQTwOaQ)


# In[33]:


df_with_clusters_classified[df_with_clusters_classified['louvain_cmty']==116]
# Spanish speakers with interest in guitar


# In[36]:


def calc_sum_conductance(column_name): 
    sum_conductance = 0
    unique_cluster_names = list(set(df_with_clusters_classified[column_name]))
    for cluster_name in unique_cluster_names: 
        S = set(df_with_clusters_classified[df_with_clusters_classified[column_name]==cluster_name]['node ID'].values)
        try: 
            conductance = (nx.cut_size(G, S)/min(nx.volume(G, S), nx.volume(G,nodes_G-S)))
        except ZeroDivisionError:
            continue
        sum_conductance += conductance
    return sum_conductance

cnm_sum_conductance = calc_sum_conductance('cnm_cmty')
print('CNM CONDUCTANCE')
print(cnm_sum_conductance)
louvain_sum_conductance = calc_sum_conductance('louvain_cmty')
print('LOUVAIN CONDUCTANCE')
print(louvain_sum_conductance)
category_sum_conductance = calc_sum_conductance('category')
print('CATEOGRY CONDUCTANCE')
print(category_sum_conductance)


def conductance(g, coms):
    ms = len(coms.edges())
    edges_outside = 0
    for n in coms.nodes():
        neighbors = g.neighbors(n)
        for n1 in neighbors:
            if n1 not in coms:
                edges_outside += 1
    try:
        ratio = float(edges_outside) / ((2 * ms) + edges_outside)
    except:
        return 0
    return ratio


def normalized_cut(g, coms):
    ms = len(coms.edges())
    edges_outside = 0
    for n in coms.nodes():
        neighbors = g.neighbors(n)
        for n1 in neighbors:
            if n1 not in coms:
                edges_outside += 1
    try:
        ratio = (float(edges_outside) / ((2 * ms) + edges_outside)) + float(edges_outside) / (2 * (len(g.edges()) - ms) + edges_outside)
    except:
        return 0

    return ratio


# In[ ]:


def calc_sum_triangle_participation_ratio(column_name): 
    # fraction of nodes in S that belong to a triad 
    sum_conductance = 0
    unique_cluster_names = list(set(df_with_clusters_classified[column_name]))
    for cluster_name in unique_cluster_names: 
        S = set(df_with_clusters_classified[df_with_clusters_classified[column_name]==cluster_name]['node ID'].values)
        S = edges_df[edges_df['u'].isin(list(S))]
        S = S.append(edges_df[edges_df['v'].isin(list(S))], ignore_index=True)
        S = nx.from_pandas_edgelist(S, 'u', 'v')
        S = S.to_undirected()
        cls = nx.triangles(S)
        nc = [n for n in cls if cls[n] > 0]
        try:
            triangle_participation_ratio = float(len(nc))/len(coms)
        except:
            continue
        sum_triangle_participation_ratio += triangle_participation_ratio
    return sum_triangle_participation_ratio


# In[ ]:


# same numbers of normalized cu t
# instead compute theoretical lower bounds on the conductance community-quality score in Section 5
## Empirical Comparison of Algorithms for Network Community Detection

## The minimum normalized cut is in fact the conductance of a weighted graph. 

def calc_sum_ncut(column_name): 
    sum_ncut = 0
    unique_cluster_names = list(set(df_with_clusters_classified[column_name]))
    for cluster_name in unique_cluster_names: 
        S = set(df_with_clusters_classified[df_with_clusters_classified[column_name]==cluster_name]['node ID'].values)
        try: 
            ncut = (nx.cut_size(G, S)/nx.volume(G, S))
        except ZeroDivisionError:
            continue
        sum_ncut += ncut
    return sum_ncut

