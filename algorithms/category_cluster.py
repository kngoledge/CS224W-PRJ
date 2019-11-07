import snap
import pandas as pd
import numpy as np
from os import path
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

def data2dag(data, num_nodes):
  dag = snap.TNGraph.New()
  for i in range(num_nodes):
    dag.AddNode(i)

  for i in range(data.shape[0]):
    dag.AddEdge(int(data[i][0]), int(data[i][1]))
  FOut = snap.TFOut("../data/youtube.graph")
  dag.Save(FOut)
  return dag

def main():

  # Load data
  nodes = pd.read_csv("../data/nodes.csv", sep='\t', index_col=0)

  # Data in nice form
  headers = list(nodes.columns)
  nodes = np.asarray(nodes)

  # Load social network accordingly
  if path.exists("../data/youtube.graph"):
    FIn = snap.TFIn("../data/youtube.graph")
    social_network = snap.TNGraph.Load(FIn)
  else:
    edges = pd.read_csv("../data/edges.csv", sep='\t', index_col=0)
    edges = np.asarray(edges).astype(int)
    social_network = data2dag(edges, nodes.shape[0])

  # Find number of unique categories
  upload_col = headers.index('category')
  categories = set()
  for i in range(nodes.shape[0]):
    categories.add(nodes[i][upload_col])
  idx_to_categories = list(categories)
  categories_to_idx = dict()
  for i in range(len(idx_to_categories)):
    categories_to_idx[idx_to_categories[i]] = i

  # mapping
  node_to_category = [categories_to_idx[x] for x in nodes[:,upload_col]]
  node_to_category = np.asarray(node_to_category).astype(int)

  # category communities
  num_cmtys = len(idx_to_categories)
  num_edges = social_network.GetEdges()
  cmtys = [[] for _ in range(num_cmtys)]
  for i in range(len(node_to_category)):
    cmtys[node_to_category[i]].append(i)

  # Calculate modularity
  modularity = 0
  for cmty in cmtys:
    Nodes = snap.TIntV()
    for elem in cmty:
      Nodes.Add(int(elem))
    modularity += snap.GetModularity(social_network, Nodes, num_edges)
  print("Results from Category Clusters:")
  print("Modularity:",modularity)
  print("Number of clusters:",num_cmtys)




if __name__ == '__main__':
    main()
